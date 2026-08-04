import argparse
import gc
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import einops
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import PVSGRelationDataset
from models.tfn import (
    ClassEmbeddingModel,
    FusionModel,
    ObjectEncoder,
    PairProposalNetwork,
    TFN,
    concatenate_sub_obj,
    diversity_loss,
    get_gt_pairs,
    matched_iou_loss,
    matched_mask_loss,
    rew_bce_loss,
    zlpr_loss,
)
from tools.run_eval import evaluate
from utils.relation_matching import PVSGRelationAnnotation
from utils.gpu import preallocate_cuda_memory


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(config, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def fix_random_seeds(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def is_empty_tensor(tensor):
    return tensor.numel() == 0



def apply_runtime_overrides(config, args):
    experiment_cfg = config.setdefault("experiment", {})
    train_cfg = config.setdefault("train", {})

    if args.exp_name:
        experiment_cfg["name"] = args.exp_name

    if args.save_root or args.exp_name or "save_dir" not in train_cfg:
        save_root = args.save_root or train_cfg.get("save_root") or "outputs/results"
        exp_name = experiment_cfg.get("name")
        save_dir = os.path.join(save_root, exp_name) if exp_name else save_root
        train_cfg["save_root"] = save_root
        train_cfg["save_dir"] = save_dir
    else:
        save_dir = train_cfg["save_dir"]
        train_cfg.setdefault("save_root", os.path.dirname(save_dir) or save_dir)

    runtime_cfg = config.setdefault("runtime", {})
    runtime_cfg["config_path"] = args.config
    runtime_cfg["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    runtime_cfg["resolved_save_dir"] = save_dir
    return config


def build_dataloaders(config):
    data_cfg = config['data']
    anno_file = os.path.join(data_cfg['data_dir'], 'pvsg.json')

    train_dataset = PVSGRelationDataset(
        anno_file,
        split=data_cfg.get('train_split', 'train'),
        work_dir=data_cfg['train_features'],
        return_cid=True,
        gt_time=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )

    val_dataset = PVSGRelationDataset(
        anno_file,
        split=data_cfg.get('val_split', 'val'),
        work_dir=data_cfg['val_features'],
        return_cid=True,
        return_mask=True,
        gt_time=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_annotations = PVSGRelationAnnotation(anno_file, data_cfg.get('val_split', 'val'))
    return train_dataset, train_loader, val_dataset, val_loader, val_annotations


def build_models(config, device):
    model_cfg = config['model']
    feature_dim = model_cfg.get('feature_dim', 256)
    hidden_dim = model_cfg.get('hidden_dim', 1024)
    embed_dim = model_cfg.get('embed_dim', 256)
    num_categories = model_cfg.get('num_categories', 126)

    models = {
        'subject_encoder': ObjectEncoder(feature_dim=feature_dim).to(device),
        'object_encoder': ObjectEncoder(feature_dim=feature_dim).to(device),
        'pair_proposal_model': PairProposalNetwork(feature_dim, hidden_dim).to(device),
        'embedding_model': ClassEmbeddingModel(
            num_categories=num_categories,
            embed_dim=embed_dim,
            device=device,
        ).to(device),
        'fusion_model': FusionModel(dim=embed_dim, num_heads=model_cfg.get('fusion_heads', 8)).to(device),
        'tfn_model': TFN(model_cfg['config']).to(device),
    }
    return models


def build_optimizer(models, config):
    train_cfg = config['train']
    params = []
    for model in models.values():
        params.extend(model.parameters())
    return optim.AdamW(
        params,
        lr=train_cfg.get('lr', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )


def compute_relation_count(data_loader, num_relations):
    relation_count = torch.ones(num_relations)
    for relation_dict in tqdm(data_loader, desc='Counting relations'):
        for relation in relation_dict['relations']:
            relation_count[int(relation['relation'].item())] += len(relation['relation_span'])
    return relation_count


def sample_objects(feats, class_embeds, gt_relations, max_samples):
    if feats.size(0) <= max_samples:
        return feats, class_embeds, gt_relations

    unique_indices = set()
    for relation in gt_relations:
        unique_indices.add(relation['subject_index'].item())
        unique_indices.add(relation['object_index'].item())

    if len(unique_indices) > max_samples:
        unique_indices = set(random.sample(list(unique_indices), max_samples))

    remaining_slots = max_samples - len(unique_indices)
    all_indices = set(range(feats.size(0)))
    remaining_indices = list(all_indices - unique_indices)
    selected_indices = list(unique_indices) + random.sample(
        remaining_indices,
        min(remaining_slots, len(remaining_indices)),
    )

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
    updated_relations = []
    for relation in gt_relations:
        subject_index = relation['subject_index'].item()
        object_index = relation['object_index'].item()
        if subject_index not in index_map or object_index not in index_map:
            continue
        updated_relation = deepcopy(relation)
        updated_relation['subject_index'] = torch.tensor([index_map[subject_index]])
        updated_relation['object_index'] = torch.tensor([index_map[object_index]])
        updated_relations.append(updated_relation)

    return feats[selected_indices], class_embeds[selected_indices], updated_relations


def sample_time_window(feats, class_embeds, max_frame_length):
    if feats.size(1) <= max_frame_length:
        custom_span = [0, feats.size(1)]
    else:
        start = random.randint(0, feats.size(1) - max_frame_length)
        custom_span = [start, start + max_frame_length]
    return (
        feats[:, custom_span[0]:custom_span[1], :],
        class_embeds[:, custom_span[0]:custom_span[1], :],
        custom_span,
    )


def save_checkpoint(models, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dicts = {name: model.state_dict() for name, model in models.items()}
    torch.save(state_dicts, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))


def set_train_mode(models):
    for model in models.values():
        model.train()


def set_eval_mode(models):
    for model in models.values():
        model.eval()


class ExperimentLogger:
    def __init__(self, save_dir, config):
        self.save_dir = save_dir
        self.train_log_path = os.path.join(save_dir, "train.log")
        self.swanlab = None

        logging_cfg = config.get("logging", {})
        if logging_cfg.get("use_swanlab", False):
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            import swanlab

            experiment_name = config.get("experiment", {}).get("name")
            swanlab.init(
                project=logging_cfg.get("project", "TFN"),
                experiment_name=experiment_name or time.strftime("%Y%m%d_%H%M%S"),
            )
            self.swanlab = swanlab

    def log(self, values, *, write_to_file=True):
        values = {key: float(value) for key, value in values.items()}
        if write_to_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = " ".join([timestamp] + [f"{key}={value:.6g}" for key, value in values.items()])
            with open(self.train_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.swanlab is not None:
            self.swanlab.log(values)


def init_logger(save_dir, config):
    return ExperimentLogger(save_dir, config)


def log_losses(logger, epoch, step, losses):
    values = {'epoch': epoch, 'step': step}
    values.update({f"loss/{name}": value for name, value in losses.items()})
    logger.log(values)

def step_optimizer(optimizer, logger, epoch, global_step, loss_sums, num_batches):
    optimizer.step()
    optimizer.zero_grad()
    global_step += 1
    if logger is not None:
        average_losses = {name: value / num_batches for name, value in loss_sums.items()}
        log_losses(logger, epoch, global_step, average_losses)
    return global_step


def train_one_epoch(
    models,
    data_loader,
    optimizer,
    relation_count,
    config,
    device,
    epoch,
    global_step,
    logger=None,
):
    train_cfg = config['train']
    max_samples = train_cfg.get('num_max_samples', 100)
    max_frame_length = train_cfg.get('max_frame_length', 900)
    accumulation_steps = train_cfg.get('accumulation_steps', 32)
    if accumulation_steps < 1:
        raise ValueError("train.accumulation_steps must be positive.")

    subject_encoder = models['subject_encoder']
    object_encoder = models['object_encoder']
    pair_proposal_model = models['pair_proposal_model']
    embedding_model = models['embedding_model']
    fusion_model = models['fusion_model']
    tfn_model = models['tfn_model']

    optimizer.zero_grad()
    loss_sums = {}
    batches_since_step = 0
    with tqdm(total=len(data_loader.dataset), ncols=120, mininterval=0.3) as pbar:
        for batch_idx, relation_dict in enumerate(data_loader):
            feats = relation_dict['feats'][0].float().to(device)    # Batch size equals 1
            gt_relations = relation_dict['relations']
            class_embeds = embedding_model(relation_dict).to(device)

            feats, class_embeds, gt_relations = sample_objects(feats, class_embeds, gt_relations, max_samples)
            feats, class_embeds, custom_span = sample_time_window(feats, class_embeds, max_frame_length)
            feats = fusion_model(feats, class_embeds)

            sub_feats = subject_encoder(feats)
            obj_feats = object_encoder(feats)
            pred_matrix = pair_proposal_model(sub_feats, obj_feats)

            gt_matrix = torch.zeros_like(pred_matrix).to(device)
            for relation in gt_relations:
                gt_matrix[relation['subject_index'], relation['object_index']] = 1

            gt_pair = einops.rearrange(gt_matrix, "sub obj -> 1 (sub obj)")
            pred_pair = einops.rearrange(pred_matrix, "sub obj -> 1 (sub obj)")
            loss_pair = zlpr_loss(gt_pair, pred_pair)

            num_top_pairs = min(
                train_cfg.get('num_top_pairs', 50),
                train_cfg.get('pair_budget', 10000) // obj_feats.shape[1],
            )
            selected_pairs = get_gt_pairs(gt_relations, num_top_pairs)
            if not selected_pairs:
                pbar.update(1)
                continue

            max_pairs = train_cfg.get('pair_sample_budget', 5400) // feats.size(1)
            crop_pairs = train_cfg.get('pair_crop_budget', 4500) // feats.size(1)
            if len(selected_pairs) > max_pairs:
                start = random.randint(0, len(selected_pairs) - crop_pairs)
                selected_pairs = selected_pairs[start:start + crop_pairs]

            concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats, selected_pairs)
            gt_spans, gt_weights, gt_probs = tfn_model.generate_gt_span(
                gt_relations,
                selected_pairs,
                concatenated_feats.shape,
                custom_span,
            )
            if all(is_empty_tensor(t) for t in gt_weights):
                pbar.update(1)
                continue
            gt_probs = tfn_model.rel_fusion(gt_spans, gt_probs)

            gauss_center, gauss_width, time_pred, gauss_weight, pred_probs = tfn_model(
                concatenated_feats,
                gt_weights=gt_weights,
            )

            iou_loss, match_idx, _ = matched_iou_loss(time_pred, gt_spans)
            div_loss = diversity_loss(gauss_weight, config.get('loss', {}).get('diversity_lambda', 0.146))
            mask_loss = matched_mask_loss(gauss_weight, gt_weights, match_idx)
            loss_prob = rew_bce_loss(gt_probs, pred_probs, relation_count.to(device))

            weights = train_cfg.get('loss_weights', {})
            loss = (
                weights.get('pair', 1.0) * loss_pair
                + weights.get('iou', 10.0) * iou_loss
                + weights.get('mask', 5.0) * mask_loss
                + weights.get('diversity', 25.0) * div_loss
                + weights.get('prob', 5.0) * loss_prob
            )
            loss = loss / accumulation_steps
            loss.backward()

            loss_values = {
                'J_all': loss.item() * accumulation_steps,
                'J_pair': loss_pair.item(),
                'J_iou': iou_loss.item(),
                'J_mask': mask_loss.item(),
                'J_div': div_loss.item(),
                'J_prob': loss_prob.item(),
            }
            for name, value in loss_values.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value
            batches_since_step += 1
            pbar.set_postfix(**loss_values)
            pbar.update(1)

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(data_loader):
                global_step = step_optimizer(
                    optimizer, logger, epoch, global_step, loss_sums, batches_since_step
                )
                loss_sums.clear()
                batches_since_step = 0

            gc.collect()

    return global_step
def run_eval(models, val_loader, val_dataset, val_annotations, config, result_csv_path, eval_tag, device, logger=None):
    eval_cfg = config['eval']
    evaluate(
        models,
        val_loader,
        val_dataset,
        val_annotations,
        device,
        data_dir=config['data']['data_dir'],
        result_csv_path=result_csv_path,
        eval_tag=eval_tag,
        num_top_pairs=eval_cfg.get('num_top_pairs', 100),
        temporal_margin=eval_cfg.get('temporal_margin', 2.0),
        logger=logger,
    )

def main():
    parser = argparse.ArgumentParser(description='Train TFN relation model')
    parser.add_argument('--config', default='configs/tfn/train_ips.yaml')
    parser.add_argument('--exp-name', default=None)
    parser.add_argument("--save-root", default=None, help="Override train.save_root before appending experiment name.")
    args = parser.parse_args()
    
    preallocate_cuda_memory()

    config = apply_runtime_overrides(load_yaml(args.config), args)

    train_cfg = config['train']
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    fix_random_seeds(train_cfg.get('seed', 2024))

    save_dir = train_cfg['save_dir']
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_yaml(config, os.path.join(save_dir, 'config.yaml'))
    result_csv_path = os.path.join(save_dir, 'result.csv')

    train_dataset, train_loader, val_dataset, val_loader, val_annotations = build_dataloaders(config)
    models = build_models(config, device)
    optimizer = build_optimizer(models, config)
    relation_count = compute_relation_count(train_loader, config['model'].get('num_relations', 57))
    logger = init_logger(save_dir, config)

    print(f'save to {save_dir}')
    print('Start Training', flush=True)

    num_epochs = train_cfg.get('epochs', 100)
    eval_cfg = config.get('eval', {})
    global_step = 0
    for epoch in range(num_epochs):
        current_epoch = epoch + 1
        set_train_mode(models)
        print(f'Epoch {current_epoch}/{num_epochs}', flush=True)
        global_step = train_one_epoch(
            models,
            train_loader,
            optimizer,
            relation_count,
            config,
            device,
            epoch=current_epoch,
            global_step=global_step,
            logger=logger,
        )
        checkpoint_cfg = config.get("checkpoint", {})
        if checkpoint_cfg.get("enabled", True) and current_epoch >= checkpoint_cfg.get("start_epoch", 1) and current_epoch % checkpoint_cfg.get("interval", 5) == 0:
            save_checkpoint(models, checkpoint_dir, current_epoch)

        if eval_cfg.get("enabled", True) and current_epoch >= eval_cfg.get("start_epoch", 20) and current_epoch % eval_cfg.get("interval", 5) == 0:
            print("Evaluation Starts...", flush=True)
            set_eval_mode(models)
            run_eval(
                models,
                val_loader,
                val_dataset,
                val_annotations,
                config,
                result_csv_path,
                f"epoch{current_epoch}",
                device,
                logger=logger,
            )

if __name__ == '__main__':
    main()

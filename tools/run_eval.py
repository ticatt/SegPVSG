import argparse
import ctypes
import gc
import os
import platform
import re
import sys
from pathlib import Path

import torch
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
    pick_top_pairs_eval,
    tfn_generate_results,
)
from utils.eval_metrics import (
    GTMaskCache,
    calculate_final_metrics,
    calculate_pair_recall_at_k,
    calculate_viou,
    save_metrics_to_csv,
)
from utils.relation_matching import PVSGRelationAnnotation
from utils.temporal import create_meanwhile_matrix, remove_not_existed_pairs, span_to_mask
from utils.gpu import preallocate_cuda_memory


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_eval_mode(models):
    for model in models.values():
        model.eval()


def release_memory():
    gc.collect()
    if platform.system() == 'Linux':
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except OSError:
            pass


def evaluate(
    models,
    data_loader,
    dataset,
    annotations,
    device,
    *,
    data_dir,
    result_csv_path,
    eval_tag,
    num_top_pairs=100,
    temporal_margin=2.0,
    logger=None,
):
    relation_names = dataset.relations
    k_values = [20, 50, 100]
    relation_recall_dict = {
        k: {
            idx: {'name': name, 'total': 0, 'hit': 0, 'weak_hit': 0}
            for idx, name in enumerate(relation_names)
        }
        for k in k_values
    }

    set_eval_mode(models)
    subject_encoder = models['subject_encoder']
    object_encoder = models['object_encoder']
    pair_proposal_model = models['pair_proposal_model']
    embedding_model = models['embedding_model']
    fusion_model = models['fusion_model']
    tfn_model = models['tfn_model']

    pair_recall_list = []
    pair_recall_all_list = []

    for relation_dict in tqdm(data_loader, total=len(dataset), ncols=120):
        with torch.no_grad():
            vid = relation_dict['vid'][0]
            feats = relation_dict['feats'][0].float().to(device)
            gt_relations = relation_dict['relations']
            n_frames = feats.size(1)

            meanwhile_matrix = create_meanwhile_matrix(feats, 5)
            class_embeds = embedding_model(relation_dict).to(device)
            feats = fusion_model(feats, class_embeds)

            sub_feats = subject_encoder(feats)
            obj_feats = object_encoder(feats)
            pred_matrix = pair_proposal_model(sub_feats, obj_feats)

            selected_pairs = pick_top_pairs_eval(pred_matrix, num_top_pairs)
            selected_pairs = remove_not_existed_pairs(meanwhile_matrix, selected_pairs)

            gt_pairs = [
                [
                    int(relation['subject_index'].item()),
                    int(relation['object_index'].item()),
                ]
                for relation in gt_relations
            ]

            pair_recall = calculate_pair_recall_at_k(selected_pairs, gt_pairs, 20)
            pair_recall_all = calculate_pair_recall_at_k(
                selected_pairs,
                gt_pairs,
                len(selected_pairs),
            )
            pair_recall_list.append(pair_recall)
            pair_recall_all_list.append(pair_recall_all)

            concatenated_feats = concatenate_sub_obj(sub_feats, obj_feats, selected_pairs)
            _, _, time_pred, pred_exist, pred_cls = tfn_model(
                concatenated_feats,
                eval=True,
            )
            results = tfn_generate_results(
                time_pred,
                pred_cls,
                pred_exist,
                selected_pairs,
                temporal_margin,
                average_score=True,
            )

            gt_dict = annotations[vid]
            gt_object_dict = {
                object_dict['object_id']: object_dict['category']
                for object_dict in gt_dict['objects']
            }
            pred_obj_dict_with_mask = {
                idx: mask_dict
                for idx, mask_dict in enumerate(relation_dict['masks'])
            }
            gt_mask_cache = GTMaskCache(vid, data_dir)

            for gt_relation in gt_dict['relations']:
                sub_idx, obj_idx, rel_idx, gt_span_list = gt_relation
                rel_key = (
                    int(gt_object_dict[sub_idx]),
                    int(gt_object_dict[obj_idx]),
                    int(rel_idx),
                )

                for k in k_values:
                    relation_recall_dict[k][rel_key[2]]['total'] += 1

                for idx, result in enumerate(results[:k_values[-1]]):
                    subject_mask = pred_obj_dict_with_mask[result['subject_index']]
                    object_mask = pred_obj_dict_with_mask[result['object_index']]
                    if len(subject_mask) == 0 or len(object_mask) == 0:
                        continue

                    pred_key = (
                        int(subject_mask['cid'][0]),
                        int(object_mask['cid'][0]),
                        result['relation'],
                    )
                    if pred_key != rel_key:
                        continue

                    time_mask = span_to_mask(result['relation_span'], n_frames)
                    iou, _ = calculate_viou(
                        (sub_idx, obj_idx, gt_span_list),
                        (
                            [subject_mask['rle']],
                            [object_mask['rle']],
                            time_mask,
                        ),
                        gt_mask_cache,
                    )

                    if iou >= 0.1:
                        for k in k_values:
                            if idx < k:
                                relation_recall_dict[k][rel_key[2]]['weak_hit'] += 1

                    if iou >= 0.5:
                        for k in k_values:
                            if idx < k:
                                relation_recall_dict[k][rel_key[2]]['hit'] += 1
                        break

            del relation_dict
            del pred_obj_dict_with_mask
            del gt_mask_cache
            del feats, sub_feats, obj_feats, pred_matrix
            del concatenated_feats, time_pred, pred_exist, pred_cls
        release_memory()

    final_metrics = calculate_final_metrics(relation_recall_dict, k_values)

    for k in k_values:
        print('-------------------------------------------------------------------')
        print(f"Recall@{k}: {100 * final_metrics[k]['recall']:.2f}")
        print(f"Mean Recall@{k}: {100 * final_metrics[k]['mean_recall']:.2f}")
        print(f"Weak Recall@{k}: {100 * final_metrics[k]['weak_recall']:.2f}")
        print(f"Weak Mean Recall@{k}: {100 * final_metrics[k]['weak_mean_recall']:.2f}")
        print('-------------------------------------------------------------------')

    save_metrics_to_csv(
        final_metrics,
        pair_recall_list,
        k_values,
        result_csv_path,
        eval_tag,
        pair_recall_all_list,
    )

    if logger:
        for k in k_values:
            logger.log({
                f"metrics/Recall@{k}": round(100 * final_metrics[k]['recall'], 2),
                f"metrics/Mean Recall@{k}": round(100 * final_metrics[k]['mean_recall'], 2),
                f"metrics/Weak Recall@{k}": round(100 * final_metrics[k]['weak_recall'], 2),
                f"metrics/Weak Mean Recall@{k}": round(100 * final_metrics[k]['weak_mean_recall'], 2),
            }, write_to_file=False)

    return relation_recall_dict

def resolve_config(args):
    if args.config:
        config_path = args.config
    else:
        if not args.exp_name:
            raise ValueError('Either --config or --exp-name must be provided.')
        config_path = os.path.join(args.save_root, args.exp_name, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')

    config = load_yaml(config_path)
    if args.exp_name:
        config.setdefault('experiment', {})['name'] = args.exp_name
    return config, config_path


def resolve_save_dir(config, args):
    train_cfg = config.setdefault('train', {})
    if args.save_dir:
        save_dir = args.save_dir
    elif train_cfg.get('save_dir'):
        save_dir = train_cfg['save_dir']
    else:
        save_root = args.save_root or train_cfg.get('save_root') or 'outputs/results'
        exp_name = config.get('experiment', {}).get('name')
        if not exp_name:
            raise ValueError('experiment.name is required to resolve save_dir.')
        save_dir = os.path.join(save_root, exp_name)

    train_cfg['save_dir'] = save_dir
    train_cfg.setdefault('save_root', os.path.dirname(save_dir) or save_dir)
    return save_dir


def checkpoint_sort_key(path):
    match = re.search(r'epoch_(\d+)\.pth$', path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(save_dir, args):
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.epoch is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'{args.epoch}.pth')
    else:
        checkpoints = sorted(Path(checkpoint_dir).glob('epoch_*.pth'), key=checkpoint_sort_key)
        if not checkpoints:
            raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {checkpoint_dir}")
        checkpoint_path = str(checkpoints[-1])

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    return checkpoint_path


def build_eval_data(config):
    data_cfg = config['data']
    data_dir = data_cfg['data_dir']
    split = data_cfg.get('val_split', 'val')
    anno_file = os.path.join(data_dir, 'pvsg.json')

    dataset = PVSGRelationDataset(
        anno_file,
        split,
        data_cfg['val_features'],
        return_mask=True,
        return_cid=True,
        gt_time=False,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    ann_dataset = PVSGRelationAnnotation(anno_file, split)
    return dataset, data_loader, ann_dataset


def build_models(config, checkpoint_path, device):
    model_cfg = config['model']
    feature_dim = model_cfg.get('feature_dim', 256)
    hidden_dim = model_cfg.get('hidden_dim', 1024)
    embed_dim = model_cfg.get('embed_dim', 256)
    num_categories = model_cfg.get('num_categories', 126)

    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    for name, model in models.items():
        model.load_state_dict(checkpoint[name])
    set_eval_mode(models)
    return models



def run_standalone_eval(config, save_dir, checkpoint_path, args):
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    dataset, data_loader, annotations = build_eval_data(config)
    models = build_models(config, checkpoint_path, device)

    eval_cfg = config.get('eval', {})
    eval_tag = args.eval_tag or Path(checkpoint_path).stem
    evaluate(
        models,
        data_loader,
        dataset,
        annotations,
        device,
        data_dir=config['data']['data_dir'],
        result_csv_path=os.path.join(save_dir, 'result.csv'),
        eval_tag=eval_tag,
        num_top_pairs=eval_cfg.get('num_top_pairs', 100),
        temporal_margin=eval_cfg.get('temporal_margin', 2.0),
    )
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TFN relation model')
    parser.add_argument('--config', default=None)
    parser.add_argument('--exp-name', default=None)
    parser.add_argument('--save-root', default='outputs/results')
    parser.add_argument("--save-dir", default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--epoch', default=None)
    parser.add_argument('--eval-tag', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    preallocate_cuda_memory()
    
    config, config_path = resolve_config(args)
    save_dir = resolve_save_dir(config, args)
    checkpoint_path = resolve_checkpoint(save_dir, args)

    print(f'config: {config_path}')
    print(f'checkpoint: {checkpoint_path}')
    print(f'eval output: {save_dir}')
    run_standalone_eval(config, save_dir, checkpoint_path, args)


if __name__ == '__main__':
    main()

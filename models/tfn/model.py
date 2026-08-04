import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import temporal_iou
from .transformer import TransformerDecoder


def _generate_mask(x, x_len):
    mask = []
    for length in x_len:
        item = torch.zeros([x.size(1)], dtype=torch.uint8, device=x.device)
        item[:int(length.item())] = 1
        mask.append(item)
    return torch.stack(mask, 0)


class RelationExistenceHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim // 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return torch.max(x, dim=1).values


class RelationClassifier(nn.Module):
    def __init__(self, input_dim, num_relations):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
        )
        self.output = nn.Linear(input_dim // 4, num_relations)

    def forward(self, x, weights=None):
        x = self.hidden(x)
        x = self.output(x)
        if weights is not None:
            weights = einops.rearrange(weights, "item frame -> item frame 1")
            x = x * weights
            x = x.sum(dim=1)
        else:
            x = torch.max(x, dim=1).values
        return x


class TemporalPredictor(nn.Module):
    def __init__(self, num_decoder_layers, d_model, num_heads, dropout):
        super().__init__()
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dropout)
        self.cross_att = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

    def forward(self, pair_feat, pair_mask, time_query):
        out, _ = self.decoder(None, None, pair_feat, pair_mask)
        time_query = einops.rearrange(time_query, "bs prop feat -> prop bs feat")
        out = einops.rearrange(out, "bs frame feat -> frame bs feat")
        h, _ = self.cross_att(time_query, out, out)
        return einops.rearrange(h, "prop bs feat -> bs prop feat")


class TemporalAttention(nn.Module):
    def __init__(self, num_decoder_layers, d_model, num_heads, dropout):
        super().__init__()
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, dropout)

    def forward(self, pair_feat, pair_mask, gauss_weight=None):
        return self.decoder(None, None, pair_feat, pair_mask, tgt_gauss_weight=gauss_weight)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = einops.rearrange(torch.arange(max_len), "frame -> frame 1")
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class TFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config["dropout"]
        self.num_props = config["num_props"]
        self.sigma = config["sigma"]
        self.num_relations = config["num_relations"]
        self.time_query = nn.Parameter(
            torch.zeros(self.num_props, config["hidden_size"], requires_grad=True)
        ).float()
        self.vis_fc = nn.Linear(config["pair_input_size"], config["hidden_size"])
        self.positional_encoding = PositionalEncoding(config["hidden_size"])
        self.time_pred = TemporalPredictor(**config["Temporal_Predictor"])
        self.time_att = TemporalAttention(**config["Temporal_Attention"])
        self.fc_gauss = nn.Linear(config["hidden_size"], self.num_props * 2)
        self.center_head = nn.Sequential(nn.Linear(config["hidden_size"], 1), nn.Sigmoid())
        self.width_head = nn.Sequential(nn.Linear(config["hidden_size"], 1), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(config["hidden_size"])
        self.exist_head = RelationExistenceHead(config["hidden_size"])
        self.cls_head = RelationClassifier(config["hidden_size"], config["num_relations"])

    def gen_gaussian_masks(self, pair_feat, pair_mask, time_query):
        pair_feat = einops.rearrange(pair_feat, "bs frame feat -> frame bs feat")
        pair_feat = self.positional_encoding(pair_feat)
        pair_feat = einops.rearrange(pair_feat, "frame bs feat -> bs frame feat")
        h = self.time_pred(pair_feat, pair_mask, time_query)
        gauss_center = self.center_head(h).squeeze(-1)
        gauss_width = self.width_head(h).squeeze(-1)
        gauss_center = einops.rearrange(gauss_center, "bs prop -> (bs prop)")
        gauss_width = einops.rearrange(gauss_width, "bs prop -> (bs prop)")
        return gauss_center, gauss_width

    def generate_gauss_weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len, device=center.device)
        weight = einops.repeat(weight, "frame -> span frame", span=center.size(0))
        center = einops.rearrange(center, "span -> span 1")
        width = einops.rearrange(width, "span -> span 1").clamp(1e-2) / self.sigma
        weight = 0.3989422804014327 / width * torch.exp(-((weight - center) ** 2) / (2 * width**2))
        return weight / weight.max(dim=-1, keepdim=True)[0]

    def gen_mask_boundary(self, center, width):
        left = torch.clamp(center - width / 2, min=0)
        right = torch.clamp(center + width / 2, max=1)
        return left, right

    def concat_gt_and_feats(self, gt_weights, pair_feats):
        concat_gt = torch.cat(gt_weights, dim=0)
        gt_counts = [gt.shape[0] for gt in gt_weights]
        expanded_feats = []
        for idx, gt_count in enumerate(gt_counts):
            sample_feat = pair_feats[idx]
            expanded_feats.append(einops.repeat(sample_feat, "frame feat -> gt frame feat", gt=gt_count))
        return concat_gt, torch.cat(expanded_feats, dim=0)

    def forward(self, pair_feats, gt_weights=None, eval=False):
        batch_size, num_frames, _ = pair_feats.shape
        time_query = einops.repeat(self.time_query, "prop feat -> bs prop feat", bs=batch_size)
        frames_len = torch.full(size=(batch_size,), fill_value=num_frames, device=pair_feats.device)

        pair_feats = F.dropout(pair_feats, self.dropout, self.training)
        pair_feats = self.vis_fc(pair_feats)
        pair_mask = _generate_mask(pair_feats, frames_len)

        gauss_center, gauss_width = self.gen_gaussian_masks(pair_feats, pair_mask, time_query)
        gauss_weight = self.generate_gauss_weight(num_frames, gauss_center, gauss_width)
        gauss_left, gauss_right = self.gen_mask_boundary(gauss_center, gauss_width)

        gauss_left = einops.rearrange(gauss_left, "(bs prop) -> bs prop", bs=batch_size)
        gauss_right = einops.rearrange(gauss_right, "(bs prop) -> bs prop", bs=batch_size)
        time_pred = torch.stack([gauss_left, gauss_right], dim=-1)

        if eval:
            props_feats = einops.repeat(
                pair_feats,
                "bs frame feat -> (bs prop) frame feat",
                prop=self.num_props,
            )
            props_mask = einops.repeat(
                pair_mask,
                "bs frame -> (bs prop) frame",
                prop=self.num_props,
            )

            if num_frames > 500:
                outputs = []
                chunk_size = 60
                for idx in range(0, props_feats.size(0), chunk_size):
                    chunk = props_feats[idx:idx + chunk_size]
                    chunk_mask = props_mask[idx:idx + chunk_size]
                    chunk_gauss = gauss_weight[idx:idx + chunk_size]
                    out, _ = self.time_att(chunk, chunk_mask, gauss_weight=chunk_gauss)
                    outputs.append(out)
                props_feats = torch.cat(outputs, dim=0)
            else:
                props_feats, _ = self.time_att(props_feats, props_mask, gauss_weight=gauss_weight)

            props_feats = self.layer_norm(props_feats)
            pred_exist = self.exist_head(props_feats)
            pred_cls = self.cls_head(props_feats)
            return gauss_center, gauss_width, time_pred, pred_exist, pred_cls

        if gt_weights is not None:
            concat_gt, gt_pair_feats = self.concat_gt_and_feats(gt_weights, pair_feats)
            gt_feats, _ = self.time_att(gt_pair_feats, None, gauss_weight=concat_gt)
            gt_feats = self.layer_norm(gt_feats)
            gt_cls = self.cls_head(gt_feats)
        else:
            gt_cls = None

        gauss_weight = einops.rearrange(gauss_weight, "(bs prop) frame -> bs prop frame", bs=batch_size)
        return gauss_center, gauss_width, time_pred, gauss_weight, gt_cls

    def generate_gt_span(self, gt_relations, selected_pairs, shape, custom_span):
        num_relations = self.num_relations
        num_frames = shape[1]
        device = self.time_query.device
        gt_spans = [[] for _ in range(len(selected_pairs))]
        gt_masks = [torch.empty(0, device=device) for _ in range(len(selected_pairs))]
        gt_probs = [torch.empty(0, device=device) for _ in range(len(selected_pairs))]

        for relation in gt_relations:
            subject_index = relation["subject_index"].item()
            object_index = relation["object_index"].item()
            relation_index = relation["relation"].item()
            relation_span = relation["relation_span"]

            if [subject_index, object_index] not in selected_pairs:
                continue

            pair_index = selected_pairs.index([subject_index, object_index])
            center = []
            width = []
            for time_span in relation_span:
                if type(time_span[0]) != float:
                    if (custom_span[1] - time_span[0].item()) < 3 or (time_span[1].item() - custom_span[0]) < 3:
                        continue

                    time_span[0] = max((time_span[0].item() - custom_span[0]) / num_frames, 0.0)
                    time_span[1] = min((time_span[1].item() - custom_span[0]) / num_frames, 1.0)
                    center.append((time_span[0] + time_span[1]) / 2)
                    width.append(time_span[1] - time_span[0])
                    gt_spans[pair_index].append(time_span)

                gt_prob = torch.zeros(1, num_relations, device=device)
                gt_prob[0, relation_index] = 1
                gt_probs[pair_index] = torch.cat([gt_probs[pair_index], gt_prob], dim=0)

            if center:
                center = torch.tensor(center, device=device)
                width = torch.tensor(width, device=device)
                gt_mask = self.generate_gauss_weight(num_frames, center, width)
                if torch.isnan(gt_mask).any():
                    print("gt_mask is nan")
                gt_masks[pair_index] = torch.cat([gt_masks[pair_index], gt_mask], dim=0)

        return gt_spans, gt_masks, gt_probs

    def rel_fusion(self, gt_spans, gt_probs):
        for idx, gt_span in enumerate(gt_spans):
            if len(gt_span) <= 1:
                continue
            iou_matrix = temporal_iou(torch.tensor(gt_span), torch.tensor(gt_span))
            rows, cols = torch.where(iou_matrix > 0.5)
            indices_dict = {}
            synthesized_labels = gt_probs[idx].clone()
            for row, col in zip(rows.tolist(), cols.tolist()):
                indices_dict.setdefault(row, []).append(col)
            for row, cols_for_row in indices_dict.items():
                related_labels = gt_probs[idx][cols_for_row]
                synthesized_labels[row] = torch.any(related_labels, dim=0)
            gt_probs[idx] = synthesized_labels
        return torch.cat(gt_probs, dim=0)

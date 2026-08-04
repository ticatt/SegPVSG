import random

import torch


def zlpr_loss(y_true, y_pred):
    """Multi-label categorical cross-entropy used by the pair proposal head."""
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 9999
    y_pred_pos = y_pred - (1 - y_true) * 9999
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def _to_int(value):
    return int(value.item()) if hasattr(value, "item") else int(value)


def get_gt_pairs(gt_relations, num_total_pairs=100):
    gt_pairs = {
        (_to_int(relation["subject_index"]), _to_int(relation["object_index"]))
        for relation in gt_relations
    }
    gt_pairs = list(gt_pairs)
    if len(gt_pairs) > num_total_pairs:
        gt_pairs = random.sample(gt_pairs, num_total_pairs)
    return [[subject_idx, object_idx] for subject_idx, object_idx in gt_pairs]


def concatenate_sub_obj(sub_feats, obj_feats, selected_pairs):
    pair_feats = []
    for subject_idx, object_idx in selected_pairs:
        pair_feats.append(torch.cat([sub_feats[subject_idx], obj_feats[object_idx]], dim=-1))
    return torch.stack(pair_feats)

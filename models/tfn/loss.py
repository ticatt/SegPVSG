import einops
import torch
import torch.nn.functional as F


def temporal_iou(pred_spans, gt_spans):
    pred_start = einops.rearrange(pred_spans[:, 0], "pred -> 1 pred")
    pred_end = einops.rearrange(pred_spans[:, 1], "pred -> 1 pred")
    gt_start = einops.rearrange(gt_spans[:, 0], "gt -> gt 1")
    gt_end = einops.rearrange(gt_spans[:, 1], "gt -> gt 1")

    inter_start = torch.max(pred_start, gt_start)
    inter_end = torch.min(pred_end, gt_end)
    inter = torch.clamp(inter_end - inter_start, min=0)
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / (union + 1e-8)


def endpoint_distance(pred_spans, gt_spans):
    gt_start = einops.rearrange(gt_spans[:, 0], "gt -> gt 1")
    gt_end = einops.rearrange(gt_spans[:, 1], "gt -> gt 1")
    pred_start = einops.rearrange(pred_spans[:, 0], "pred -> 1 pred")
    pred_end = einops.rearrange(pred_spans[:, 1], "pred -> 1 pred")
    return torch.abs(pred_start - gt_start) + torch.abs(pred_end - gt_end)


def matched_iou_loss(pred, gt_span, eps=1e-6):
    batch_size = pred.size(0)
    total_loss = 0.0
    match_idx = []
    exist_labels = []

    for batch_idx in range(batch_size):
        pred_spans = pred[batch_idx]
        gt_spans = torch.tensor(gt_span[batch_idx], device=pred.device, dtype=torch.float)
        exist_label = torch.zeros(pred_spans.shape[0], device=pred.device)

        if len(gt_spans) == 0:
            match_idx.append(torch.tensor([-1], device=pred.device))
            exist_labels.append(exist_label)
            continue

        iou_matrix = temporal_iou(pred_spans, gt_spans)
        max_ious, max_indices = torch.max(iou_matrix, dim=1)

        zero_mask = max_ious == 0
        if torch.any(zero_mask):
            dist_matrix = endpoint_distance(pred_spans, gt_spans[zero_mask])
            max_indices[zero_mask] = torch.argmin(dist_matrix, dim=1)

        matched_ious = iou_matrix[torch.arange(len(gt_spans), device=pred.device), max_indices]
        match_idx.append(max_indices)
        total_loss += -torch.log(matched_ious + eps).mean()

        max_ious, _ = torch.max(iou_matrix, dim=0)
        exist_label[max_ious > 0.5] = 1.0
        exist_labels.append(exist_label)

    return total_loss / batch_size, match_idx, torch.stack(exist_labels, dim=0)


def matched_mask_loss(pred_weights, gt_weights, match_idx):
    total_loss = 0.0
    batch_size = pred_weights.size(0)

    for batch_idx in range(batch_size):
        pred_masks = pred_weights[batch_idx]
        gt_tensor = gt_weights[batch_idx]
        idx_list = match_idx[batch_idx]

        if gt_tensor.numel() == 0:
            continue

        matched_preds = pred_masks[idx_list]
        assert torch.all(matched_preds >= 0 - 1e-6) and torch.all(matched_preds <= 1 + 1e-6), (
            f"Predicted mask values out of [0,1] range. Min:{matched_preds.min().item()} Max:{matched_preds.max().item()}"
        )
        assert torch.all(gt_tensor >= 0 - 1e-6) and torch.all(gt_tensor <= 1 + 1e-6), (
            f"GT mask values out of [0,1] range. Min:{gt_tensor.min().item()} Max:{gt_tensor.max().item()}"
        )
        total_loss += F.mse_loss(matched_preds, gt_tensor, reduction="mean")

    return total_loss / batch_size


def diversity_loss(gauss_weight, _lambda=0.146):
    batch_size, num_props, _ = gauss_weight.shape
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = einops.rearrange(
        torch.eye(num_props, device=gauss_weight.device),
        "prop_i prop_j -> 1 prop_i prop_j",
    ) * _lambda
    source = torch.matmul(
        gauss_weight,
        einops.rearrange(gauss_weight, "bs prop frame -> bs frame prop"),
    )
    div_loss = torch.norm(target - source, dim=(1, 2)) ** 2
    return div_loss.mean()


def rew_bce_loss(y_true, y_pred, class_counts):
    total_counts = class_counts.sum()
    class_weights = total_counts / class_counts
    return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=class_weights)

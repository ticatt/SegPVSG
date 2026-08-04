import csv
import logging
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


def calculate_pair_recall_at_k(selected_pairs, gt_pairs, k=20):
    selected_pairs_set = {tuple(pair) for pair in selected_pairs[:k]}
    gt_pairs_set = {tuple(pair) for pair in gt_pairs}
    if not gt_pairs_set:
        return 0
    return len(selected_pairs_set.intersection(gt_pairs_set)) / len(gt_pairs_set)


def calculate_final_metrics(relation_recall_dict, k_values):
    final_metrics = {}
    valid_relations = [
        rel for rel in relation_recall_dict[k_values[0]].values()
        if rel["total"] != 0
    ]
    num_valid_relations = len(valid_relations)

    for k in k_values:
        total_hit = sum(rel["hit"] for rel in relation_recall_dict[k].values())
        total_weak_hit = sum(rel["weak_hit"] for rel in relation_recall_dict[k].values())
        total_gt = sum(rel["total"] for rel in relation_recall_dict[k].values())

        final_metrics[k] = {
            "recall": total_hit / total_gt if total_gt > 0 else 0,
            "mean_recall": _mean_relation_score(relation_recall_dict[k], "hit", num_valid_relations),
            "weak_recall": total_weak_hit / total_gt if total_gt > 0 else 0,
            "weak_mean_recall": _mean_relation_score(relation_recall_dict[k], "weak_hit", num_valid_relations),
        }
    return final_metrics


class GTMaskCache:
    def __init__(self, vid, data_dir):
        self.masks_root = os.path.join(
            data_dir,
            _get_data_source(vid),
            "masks",
            vid,
        )
        self.rles = {}

    def get_pair(self, frame_id, subject_id, object_id):
        subject_key = (frame_id, subject_id)
        object_key = (frame_id, object_id)
        missing_keys = [
            (key, instance_id)
            for key, instance_id in (
                (subject_key, subject_id),
                (object_key, object_id),
            )
            if key not in self.rles
        ]
        if missing_keys:
            mask_path = os.path.join(self.masks_root, f"{frame_id:04d}.png")
            with Image.open(mask_path) as image:
                pan_mask = np.asarray(image)
            for key, instance_id in missing_keys:
                binary_mask = np.asfortranarray(pan_mask == instance_id, dtype=np.uint8)
                self.rles[key] = (
                    mask_utils.encode(binary_mask)
                    if binary_mask.any()
                    else None
                )
        return self.rles[subject_key], self.rles[object_key]


def calculate_viou(gt_set, pred_set, mask_cache):
    gt_sub_idx, gt_obj_idx, gt_span_list = gt_set
    pred_sub_mask_list, pred_obj_mask_list, pred_span_list = pred_set

    pred_sub_masks = _collect_pred_masks(pred_sub_mask_list)
    pred_obj_masks = _collect_pred_masks(pred_obj_mask_list)

    gt_real_span = np.zeros_like(pred_span_list)
    pred_hit_span = np.zeros_like(pred_span_list)

    for start, end in gt_span_list:
        for frame_id in range(start, end + 1):
            if frame_id >= len(pred_span_list):
                continue

            gt_sub_rle, gt_obj_rle = mask_cache.get_pair(
                frame_id,
                gt_sub_idx,
                gt_obj_idx,
            )
            if gt_sub_rle is None or gt_obj_rle is None:
                continue

            gt_real_span[frame_id] = 1
            if frame_id in pred_sub_masks and frame_id in pred_obj_masks:
                sub_iou = _rle_iou(gt_sub_rle, pred_sub_masks[frame_id])
                obj_iou = _rle_iou(gt_obj_rle, pred_obj_masks[frame_id])
                if sub_iou >= 0.5 and obj_iou >= 0.5:
                    pred_hit_span[frame_id] = 1

    pred_hit_real = np.logical_and(pred_hit_span == 1, pred_span_list == 1).astype(pred_hit_span.dtype)
    iou_weak = calculate_iou(pred_hit_span, gt_real_span)

    iou = calculate_iou(pred_hit_real, gt_real_span)
    return iou, iou_weak


def save_metrics_to_csv(final_metrics, pair_recall_list, k_values, csv_file_path, eval_tag, pair_recall_all_list):
    file_exists = os.path.isfile(csv_file_path)
    header = ["Model", "Pair Recall"]
    for k in k_values:
        header.append(f"R/mR@{k}")
    for k in k_values:
        header.append(f"wR/wmR@{k}")

    row = [
        eval_tag,
        f"{100 * np.array(pair_recall_list).mean():.2f}/{100 * np.array(pair_recall_all_list).mean():.2f}",
    ]
    for k in k_values:
        recall = final_metrics[k]["recall"]
        mean_recall = final_metrics[k]["mean_recall"]
        row.append(f"{100 * recall:.2f}/{100 * mean_recall:.2f}")
    for k in k_values:
        weak_recall = final_metrics[k]["weak_recall"]
        weak_mean_recall = final_metrics[k]["weak_mean_recall"]
        row.append(f"{100 * weak_recall:.2f}/{100 * weak_mean_recall:.2f}")

    with open(csv_file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def calculate_iou(span1, span2):
    intersection = (span1 * span2).sum()
    union = span1.sum() + span2.sum() - intersection
    return intersection / union if union > 0 else 0


def calculate_mask_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0


def _mean_relation_score(relation_scores, key, num_valid_relations):
    if num_valid_relations == 0:
        return 0
    return sum(
        rel[key] / rel["total"]
        for rel in relation_scores.values()
        if rel["total"] != 0
    ) / num_valid_relations


def _collect_pred_masks(mask_list):
    pred_masks = {}
    for mask_dict in mask_list:
        pred_masks.update(mask_dict)
    return pred_masks


def _rle_iou(gt_rle, pred_rle):
    pred_rle = dict(pred_rle)
    counts = pred_rle["counts"]
    if isinstance(counts, (list, tuple)):
        counts = counts[0]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    pred_rle["counts"] = counts
    pred_rle["size"] = [
        int(value.item()) if hasattr(value, "item") else int(value)
        for value in pred_rle["size"]
    ]
    return mask_utils.iou([gt_rle], [pred_rle], [0])[0, 0]


def _get_data_source(vid):
    if vid.startswith("P"):
        return "epic_kitchen"
    if vid.split("_")[0].isdigit() and len(vid.split("_")[0]) == 4:
        return "vidor"
    return "ego4d"

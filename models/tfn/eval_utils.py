import einops
import numpy as np
import torch


def pick_top_pairs_eval(pred_matrix, num_total_pairs=100):
    with torch.no_grad():
        num_objects = pred_matrix.size(0)
        pred_matrix = pred_matrix.clone()
        pred_matrix[torch.eye(num_objects, device=pred_matrix.device).bool()] = float("-inf")
        pred_matrix_flat = einops.rearrange(pred_matrix, "sub obj -> (sub obj)")
        max_pairs = min(pred_matrix_flat.size(0), num_total_pairs)
        _, top_indices = torch.topk(pred_matrix_flat, max_pairs, sorted=True)
        top_pairs = [
            (torch.div(index, num_objects, rounding_mode="trunc"), index % num_objects)
            for index in top_indices
            if torch.div(index, num_objects, rounding_mode="trunc") != index % num_objects
        ]
        return [[int(subject_idx.item()), int(object_idx.item())] for subject_idx, object_idx in top_pairs]


def merge_overlapping_spans(span_list):
    if not span_list:
        return []

    sorted_spans = sorted(span_list, key=lambda span: span[0])
    merged = []
    current_start, current_end = sorted_spans[0]

    for start, end in sorted_spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append(np.array([current_start, current_end]))
            current_start, current_end = start, end

    merged.append(np.array([current_start, current_end]))
    return merged


def merge_proposals_with_margin(
    span_pred,
    prob,
    class_scores,
    batch_size,
    num_props,
    num_relations,
    margin,
    average_score=False,
):
    del num_props
    result = []
    span_pred = span_pred.cpu().numpy()
    prob = einops.rearrange(prob, "bs prop rel -> bs rel prop").cpu().numpy()
    class_scores = class_scores.cpu().numpy()
    score = np.zeros_like(class_scores)

    for batch_idx in range(batch_size):
        class_groups = {class_id: {"span": []} for class_id in range(num_relations)}
        for class_id in range(num_relations):
            class_prob = prob[batch_idx][class_id]
            valid_idx = np.where(class_prob > (class_scores[batch_idx][class_id] - margin))[0]
            score[batch_idx][class_id] = (
                class_prob[valid_idx].mean() if average_score else class_scores[batch_idx][class_id]
            )
            for idx in valid_idx:
                class_groups[class_id]["span"].append(span_pred[batch_idx][idx])

        for value in class_groups.values():
            value["span"] = merge_overlapping_spans(value["span"])
        result.append(class_groups)

    return result, torch.from_numpy(score)


def tfn_generate_results(span_pred, prob, pred_exist, selected_pairs, margin, average_score=True):
    del pred_exist
    results = []
    num_relations = prob.size(-1)
    batch_size, num_props, _ = span_pred.size()
    prob = einops.rearrange(prob, "(pair prop) rel -> pair prop rel", pair=batch_size, prop=num_props)
    class_scores = einops.reduce(prob, "bs prop rel -> bs rel", "max")

    proposals, score = merge_proposals_with_margin(
        span_pred,
        prob,
        class_scores,
        batch_size,
        num_props,
        num_relations,
        margin,
        average_score,
    )

    score_flat = einops.rearrange(score, "pair rel -> (pair rel)")
    _, sorted_indices = torch.sort(score_flat, descending=True)
    for index in sorted_indices:
        pair_idx = int(torch.div(index, num_relations, rounding_mode="trunc").item())
        relation_idx = int((index % num_relations).item())
        subject_idx, object_idx = selected_pairs[pair_idx]
        results.append(
            {
                "subject_index": subject_idx,
                "object_index": object_idx,
                "relation": relation_idx,
                "relation_span": proposals[pair_idx][relation_idx]["span"],
            }
        )

    return results

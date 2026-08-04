import numpy as np
import torch


def create_meanwhile_matrix(feats, min_overlap_frames):
    exist_matrix = (feats != 0).any(dim=-1)
    pairwise_exists = torch.logical_and(
        exist_matrix.unsqueeze(1),
        exist_matrix.unsqueeze(0),
    )
    return pairwise_exists.sum(dim=-1) > min_overlap_frames


def remove_not_existed_pairs(meanwhile_matrix, selected_pairs):
    valid_pairs = []
    for subject_idx, object_idx in selected_pairs:
        exists = meanwhile_matrix[subject_idx, object_idx]
        if bool(exists.item() if hasattr(exists, "item") else exists):
            valid_pairs.append([subject_idx, object_idx])
    return valid_pairs


def time_to_frame(span, num_frames):
    start = _to_float(span[0])
    end = _to_float(span[1])
    start_frame = int(round(start * num_frames))
    end_frame = int(round(end * num_frames))
    start_frame = max(0, min(start_frame, num_frames - 1))
    end_frame = max(0, min(end_frame, num_frames - 1))
    return start_frame, end_frame


def generate_mask(start_frame, end_frame, num_frames):
    mask = np.zeros(num_frames, dtype=np.float32)
    mask[start_frame:end_frame + 1] = 1.0
    return mask


def span_to_mask(relation_span, num_frames):
    final_mask = np.zeros(num_frames, dtype=np.float32)
    for span in relation_span:
        start_frame, end_frame = time_to_frame(span, num_frames)
        final_mask = np.logical_or(
            final_mask,
            generate_mask(start_frame, end_frame, num_frames),
        ).astype(np.float32)
    return final_mask


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)

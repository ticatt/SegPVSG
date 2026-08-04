import copy
import json
import pickle
import os
from pathlib import Path
from itertools import groupby
from collections import Counter

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image


class PVSGRelationAnnotation:
    def __init__(self, anno_file, split='train'):
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.video_ids = []
        for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
            for video_id in anno['split'][data_source][split]:
                self.video_ids.append(video_id)

        self.classes = anno['objects']['thing'] + anno['objects']['stuff']
        self.relations = anno['relations']

        self.videos = {}
        for video_anno in anno['data']:
            self.videos[video_anno['video_id']] = video_anno

    def __getitem__(self, vid):
        assert vid in self.videos
        video_info = copy.deepcopy(self.videos[vid])

        object_list, relation_list = [], []
        for object_content in video_info['objects']:
            object_content['category'] = self.classes.index(
                object_content['category'])
            object_list.append(object_content)

        for relation_content in video_info['relations']:
            if relation_content[2] in self.relations:
                relation_content[2] = self.relations.index(relation_content[2])
                relation_list.append(relation_content)

        return {
            'video_id': vid,
            'objects': object_list,
            'relations': relation_list,
            'relation_str': self.videos[vid]['relations']
        }


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def get_pred_mask_tubes_one_video(vid, work_dir):
    labels = []
    results = []

    # Read mask labels from the file
    label_path = f'{work_dir}/{vid}/quantitive/masks.txt'
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.strip().split())

    # Keep prediction masks as COCO RLE. They can be used directly by
    # pycocotools.mask.iou without materializing full bitmap arrays.
    for label in labels:
        frame_id, track_id, cid, h, w, m = label
        rle = {'size': [int(h), int(w)], 'counts': m}
        results.append(dict(fid=frame_id, tid=track_id, rle=rle, cid=cid))

    # Sort data by 'tid' key
    def key_func(k):
        return k['tid']

    results = sorted(results, key=key_func)

    # Group by tid
    masks_grp_by_tid = {}
    for key, value in groupby(results, key_func):
        masks_grp_by_tid[key] = list(value)

    # Organize masks into tubes
    pred_mask_tubes = {}
    for key in masks_grp_by_tid.keys():
        class_ids = []
        rle_by_frame = {}
        for content in masks_grp_by_tid[key]:
            rle_by_frame[int(content['fid']) - 1] = content['rle']
            class_ids.append(content['cid'])
        count = Counter(class_ids)
        tube_class, _ = count.most_common(1)[0]
        pred_mask_tubes[int(key)] = {'cid': tube_class, 'rle': rle_by_frame}

    return pred_mask_tubes


def get_pred_cid_one_video(vid, work_dir, mode='mask'):
    if mode == 'mask':
        labels = []
        results = []

        # 读取 mask 标签
        label_path = f'{work_dir}/{vid}/quantitive/masks.txt'
        with open(label_path, 'r') as f:
            for line in f:
                labels.append(line.strip().split())

        # 解码 mask 标签
        for label in labels:
            frame_id, track_id, cid, h, w, m = label
            results.append(dict(fid=frame_id, tid=track_id, cid=cid))

        # 按 tid 排序
        def key_func(k):
            return k['tid']

        results = sorted(results, key=key_func)

        # 分组并统计每组的主 cid
        cids_by_tid = {}
        for key, value in groupby(results, key_func):
            class_ids = [v['cid'] for v in value]
            count = Counter(class_ids)
            tube_class, _ = count.most_common(1)[0]
            cids_by_tid[int(key)] = tube_class

    return cids_by_tid


def calculate_iou(gt_mask, pred_mask):
    # Backward-compatible bitmap IoU helper for callers that pass arrays.
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0
    else:
        return intersection / union


def build_pred_mask_index(pred_mask_tubes):
    pred_by_class_frame = {}

    for pred_id, pred_tube in pred_mask_tubes.items():
        cid = int(pred_tube['cid'])
        for frame_id, rle in pred_tube['rle'].items():
            pred_by_class_frame.setdefault(cid, {}).setdefault(
                frame_id, []).append((pred_id, rle))

    return pred_by_class_frame


def encode_binary_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def match_and_process_gt_tubes(vid,
                                pvsg_dataset,
                                pred_mask_tubes,
                                data_dir='./data'):
    # Determine the data source
    if vid.startswith('P'):
        data_source = 'epic_kitchen'
    elif vid.split('_')[0].isdigit() and len(vid.split('_')[0]) == 4:
        data_source = 'vidor'
    else:
        data_source = 'ego4d'

    gt_masks_root_vid = os.path.join(data_dir, data_source, 'masks', vid)

    matching_dict = {}
    object_list = pvsg_dataset[vid]['objects']
    pred_by_class_frame = build_pred_mask_index(pred_mask_tubes)

    for frame_id, mask_path in enumerate(
            sorted(Path(gt_masks_root_vid).rglob('*.png'))):
        pan_mask = np.array(Image.open(mask_path))

        for object_entry in object_list:
            instance_id = object_entry['object_id']
            cid = int(object_entry['category'])
            candidates = pred_by_class_frame.get(cid, {}).get(frame_id, [])
            if not candidates:
                continue

            gt_mask = pan_mask == instance_id
            if not gt_mask.any():
                continue

            gt_rle = encode_binary_mask(gt_mask)
            pred_ids = [pred_id for pred_id, _ in candidates]
            pred_rles = [rle for _, rle in candidates]
            iscrowd = [0] * len(pred_rles)
            ious = mask_utils.iou([gt_rle], pred_rles, iscrowd)[0]

            for pred_id, iou in zip(pred_ids, ious):
                if iou <= 0.5:
                    continue

                if instance_id not in matching_dict:
                    matching_dict[instance_id] = {pred_id: [frame_id]}
                elif pred_id not in matching_dict[instance_id]:
                    matching_dict[instance_id][pred_id] = [frame_id]
                else:
                    matching_dict[instance_id][pred_id].append(frame_id)

    return matching_dict


def find_ranges(num_list):
    ranges = []
    start = num_list[0]
    for i in range(1, len(num_list)):
        if num_list[i] > num_list[i - 1] + 5:
            end = num_list[i - 1]
            ranges.append(f'{start}-{end}')
            start = num_list[i]
    # Add the last range
    ranges.append(f'{start}-{num_list[-1]}')
    return ranges


def compact_matching_dict(matching_dict):
    processed_dict = {}

    for outer_key, inner_dict in matching_dict.items():
        processed_inner = {}
        for inner_key, number_list in inner_dict.items():
            # Rule: Delete the inner key if the list has fewer than 5 numbers
            if len(number_list) < 5:
                continue

            # Rule: If there's only one inner key, convert the list to a range string
            if len(inner_dict) == 1:
                min_val, max_val = min(number_list), max(number_list)
                processed_inner[inner_key] = f'{min_val}-{max_val}'
            else:
                # Sorting the list to ensure continuity
                sorted_num_list = sorted(number_list)
                processed_inner[inner_key] = find_ranges(sorted_num_list)

        if processed_inner:
            processed_dict[outer_key] = processed_inner

    return processed_dict


def translate_gt_relations(matching_dict, gt_relations, gt_time=False):
    translated_relations = []

    def time_overlap(range1, range2):
        # This function checks if two ranges overlap and returns the overlapping range
        return [max(range1[0], range2[0]), min(range1[1], range2[1])]

    def is_valid_range(range1):
        # This function checks if the start of the range is less than the end
        return range1[0] < range1[1]

    def merge_sublists(lst):
        merged_list = []
        temp_dict = {}
        for sublist in lst:
            # Extract the key (first three items) and value (fourth item)
            key = tuple(sublist[:-1])
            value = sublist[-1]

            # If key is already in dictionary, append the value to the existing entry
            if key in temp_dict:
                temp_dict[key].append(value)
            else:
                # Otherwise, create a new entry with this value in a list
                temp_dict[key] = [value]

        # Convert the dictionary back into a list with merged items
        for key, values in temp_dict.items():
            merged_list.append(list(key) + [values])

        return merged_list

    def range_len(range):
        return max(0, range[1] - range[0])


    for relation in gt_relations:
        tube_1, tube_2, label, time_ranges = relation
        if tube_1 not in matching_dict or tube_2 not in matching_dict:
            continue
        tube_1_ranges = matching_dict[tube_1]
        tube_2_ranges = matching_dict[tube_2]


        if gt_time:
            for time_range in time_ranges:
                for inner_key_1, ranges_1 in tube_1_ranges.items():
                    if isinstance(ranges_1, str):  # convert string range to list
                        ranges_1 = [ranges_1]
                    range1s = []
                    for range_str_1 in ranges_1:
                        start_1, end_1 = map(int, range_str_1.split('-'))
                        range1s.append([start_1, end_1])
                    for inner_key_2, ranges_2 in tube_2_ranges.items():
                        if isinstance(ranges_2, str): 
                            ranges_2 = [ranges_2]
                        range2s = []
                        for range_str_2 in ranges_2:
                            start_2, end_2 = map(int, range_str_2.split('-'))
                            range2s.append([start_2, end_2])
                        time_len = 0
                        for range1 in range1s:
                            for range2 in range2s:
                                overlap_1 = time_overlap(time_range,
                                                        range1)
                                overlap_2 = time_overlap(time_range,
                                                        range2)
                                overlap_both = time_overlap(overlap_1, overlap_2)
                                if is_valid_range(overlap_both):
                                    time_len += range_len(overlap_both)
                        if time_len >= 3:
                            translated_relations.append([
                                    inner_key_1, inner_key_2, label,
                                    time_range
                                ])    
        else:
            for time_range in time_ranges:
                for inner_key_1, ranges_1 in tube_1_ranges.items():
                    if isinstance(ranges_1, str):  # convert string range to list
                        ranges_1 = [ranges_1]
                    for range_str_1 in ranges_1:
                        start_1, end_1 = map(int, range_str_1.split('-'))
                        for inner_key_2, ranges_2 in tube_2_ranges.items():
                            if isinstance(ranges_2,
                                        str):  # convert string range to list
                                ranges_2 = [ranges_2]
                            for range_str_2 in ranges_2:
                                start_2, end_2 = map(int, range_str_2.split('-'))
                                overlap_1 = time_overlap(time_range,
                                                        [start_1, end_1 + 1])
                                overlap_2 = time_overlap(time_range,
                                                        [start_2, end_2 + 1])
                                overlap_both = time_overlap(overlap_1, overlap_2)
                                # Check if there is an overlap and the overlap is valid
                                if is_valid_range(overlap_both):
                                    # Append the overlap, inner keys, and label to the translated relations
                                    translated_relations.append([
                                        inner_key_1, inner_key_2, label,
                                        overlap_both
                                    ])                

    return merge_sublists(translated_relations)


def process_feats(pred_feat_tubes, d=256):
    video_length = len(pred_feat_tubes[list(pred_feat_tubes.keys())[0]])
    output_list = {}
    for tube_id in pred_feat_tubes.keys():
        new_feat_tube = np.zeros([video_length, d])
        for frame_id in range(video_length):
            if pred_feat_tubes[tube_id][frame_id] is not None:
                new_feat_tube[frame_id] = pred_feat_tubes[tube_id][frame_id]['query_feat']

        output_list[tube_id] = new_feat_tube
    return output_list


def process_feats_and_relations_gt(pred_relations, pred_feat_tubes, d=256):

    output_list = []

    for item in pred_relations:
        tube_s_index, tube_o_index, relation, time_span = item

        # ignore those without long relation span
        output_dict = {
            'subject_index': tube_s_index,
            'object_index': tube_o_index,
            'relation': relation,
            'relation_span': time_span,
        }

        output_list.append(output_dict)

    return {'feats': process_feats(pred_feat_tubes), 'relations': output_list}


def process_feats_and_relations(pred_relations, pred_feat_tubes, d=256):

    output_list = []

    for item in pred_relations:
        tube_s_index, tube_o_index, relation, time_span = item
        video_length = len(pred_feat_tubes[list(pred_feat_tubes.keys())[0]])

        relation_span = np.zeros(video_length)
        for span_range in time_span:
            for i in range(span_range[0], span_range[1]):
                relation_span[i] = 1

        # processing subject feature
        for frame_id in range(video_length):
            if pred_feat_tubes[tube_s_index][frame_id] is None:
                relation_span[frame_id] = 0

        # processing object feature
        for frame_id in range(video_length):
            if pred_feat_tubes[tube_o_index][frame_id] is None:
                relation_span[frame_id] = 0

        # ignore those without long relation span
        if sum(relation_span) >= 3:
            output_dict = {
                'subject_index': tube_s_index,
                'object_index': tube_o_index,
                'relation': relation,
                'relation_span': relation_span,
            }

            output_list.append(output_dict)

    return {'feats': process_feats(pred_feat_tubes), 'relations': output_list}

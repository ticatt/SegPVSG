import pickle
import os
import json

import numpy as np

from utils.relation_matching import get_pred_mask_tubes_one_video


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class PVSGRelationDataset:
    def __init__(self,
                anno_file,
                split='train',
                work_dir='./work_dirs/train_save_qf_1106',
                return_mask=False,
                return_cid=False, 
                gt_time = False):
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.video_ids = []
        for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
            for video_id in anno['split'][data_source][split]:
                self.video_ids.append(video_id)    

        self.work_dir = work_dir
        self.split = split
        self.classes = anno['objects']['thing'] + anno['objects']['stuff']
        self.relations = anno['relations']
        self.return_mask = return_mask
        self.return_cid = return_cid
        self.gt_time = gt_time

        self.videos = {}
        for video_anno in anno['data']:
            self.videos[video_anno['video_id']] = video_anno

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        if self.gt_time:
            relation_dict = load_pickle(
                os.path.join(self.work_dir, vid, 'relations_gt_time.pickle'))
        else:
            relation_dict = load_pickle(
                os.path.join(self.work_dir, vid, 'relations.pickle'))
        relation_dict['vid'] = vid
        # getting all object features
        feat_list = []
        mapping_dict = {}
        for idx, key in enumerate(relation_dict['feats']):
            feat_list.append(relation_dict['feats'][key])
            mapping_dict[key] = idx
            
        num_items = len(feat_list)
        feat_shape = feat_list[0].shape  # 比如 (2048,)
        feat_dtype = feat_list[0].dtype  # 比如 float64
        relation_dict['feats'] = np.zeros((num_items, *feat_shape), dtype=feat_dtype)
        for i, feat in enumerate(feat_list):
            relation_dict['feats'][i] = feat

        # getting relation info
        pair_list = []
        for relation in relation_dict['relations']:
            relation['subject_index'] = mapping_dict[relation['subject_index']]
            relation['object_index'] = mapping_dict[relation['object_index']]
            pair_list.append(
                [relation['subject_index'], relation['object_index']])

        # getting pair info
        relation_dict['pairs'] = pair_list

        if self.return_mask or self.return_cid:
            rev_mapping_dict = {v: k for k, v in mapping_dict.items()}
            relation_dict['idx2key'] = rev_mapping_dict
            pred_mask_tubes = get_pred_mask_tubes_one_video(vid, self.work_dir)

            if self.return_mask:
                relation_dict['masks'] = [
                    pred_mask_tubes.get(rev_mapping_dict[idx], {})
                    for idx in range(len(rev_mapping_dict))
                ]

            if self.return_cid:
                relation_dict['cids'] = [
                    {'cid': pred_mask_tubes[key]['cid']}
                    if key in pred_mask_tubes else {}
                    for key in (rev_mapping_dict[idx] for idx in range(len(rev_mapping_dict)))
                ]
            
        return relation_dict
    
    def get_relation_info(self, index):
        """
        专门用于快速统计 relation 的轻量级方法。
        跳过了耗时的视觉特征加载 (np.array 转换)。
        """
        vid = self.video_ids[index]
        
        # 1. 加载 Pickle (和 __getitem__ 一样)
        if self.gt_time:
            path = os.path.join(self.work_dir, vid, 'relations_gt_time.pickle')
        else:
            path = os.path.join(self.work_dir, vid, 'relations.pickle')
            
        relation_dict = load_pickle(path)
        
        # 2. 【关键优化】只构建映射，不读数据！
        # 原代码是要 loop一遍 append data，还要 np.array()，这步耗时 80s
        # 现在我们只用 0.001s 构建字典
        mapping_dict = {key: idx for idx, key in enumerate(relation_dict['feats'])}
        
        # 3. 处理 Relations (这部分逻辑保留，用于统计)
        pair_list = []
        valid_relations = [] # 过滤后的 relations
        
        for relation in relation_dict['relations']:
            # 确保主体客体都在 feats 里 (虽然一般都在)
            s_key = relation['subject_index']
            o_key = relation['object_index']
            
            if s_key in mapping_dict and o_key in mapping_dict:
                # 只保留统计需要的字段，其他丢弃以节省内存
                new_rel = {
                    'relation': relation['relation'],
                    'relation_span': relation['relation_span'] # 只需要统计长度
                }
                valid_relations.append(new_rel)

        return valid_relations
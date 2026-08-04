# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import cv2
import pickle
import numpy as np
import os.path as osp
import pycocotools.mask as mask_utils
from models.unitrack.utils.log import logger
from models.unitrack.utils.meter import Timer
from models.unitrack.utils import visualize as vis
from models.unitrack.utils import io as io

class SimpleTracker(object):
    def __init__(self, track_id, qf_tube):
        self.qf_tube = []
        self.track_id = track_id
        self.qf_tube = qf_tube

def _to_numpy_feature(feat):
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if torch.is_tensor(feat):
        return feat.to(dtype=torch.float32).cpu().numpy()
    return np.asarray(feat, dtype=np.float32)


def _pool_mask_features(pan_mask, object_ids, mask_features):
    mask_features_tensor = torch.from_numpy(mask_features).float()
    if mask_features_tensor.ndim == 4:
        mask_features_tensor = mask_features_tensor.squeeze(0)

    masks = np.stack([
        pan_mask == object_id
        for object_id in object_ids
    ]).astype(np.float32)
    masks = torch.from_numpy(masks).unsqueeze(1)
    masks = torch.nn.functional.interpolate(
        masks,
        size=mask_features_tensor.shape[-2:],
        mode='nearest',
    ).squeeze(1)

    denom = masks.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    pooled_feats = torch.einsum(
        'nhw,chw->nc', masks, mask_features_tensor) / denom.squeeze(-1)
    return pooled_feats.cpu().numpy()


class VPSSequenceWriter(object):
    # Accumulate one VPS frame at a time without retaining model outputs.
    def __init__(self, save_root, feature_type='query'):
        self.save_root = save_root
        self.feature_type = feature_type
        self.timer = Timer()
        self.frame_id = 0
        self.object_id_to_tid = {}
        self.query_feat_tubes = []

        result_filename = osp.join(save_root, 'quantitive/masks.txt')
        os.makedirs(osp.dirname(result_filename), exist_ok=True)
        self.result_file = open(result_filename, 'w')

    def process(self, output):
        frame_id = self.frame_id
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, self.timer.average_time)))

        online_ids = []
        online_masks = []
        if len(output['query_feats']) > 0:
            self.timer.tic()
            pan_results = output['pan_results']
            mask_features = output.get('mask_features')
            if self.feature_type == 'mask' and mask_features is None:
                raise KeyError(
                    'mask_features not found in VPS output. '
                    'Please set feature_type="mask" before single_gpu_test.')

            frame_items = list(output['query_feats'].items())
            frame_ins_ids = [ins_id for ins_id, _ in frame_items]
            if self.feature_type == 'mask':
                frame_mask_feats = _pool_mask_features(
                    pan_results, frame_ins_ids, mask_features)
            else:
                frame_mask_feats = None

            for item_idx, (ins_id, feat) in enumerate(frame_items):
                if ins_id not in self.object_id_to_tid:
                    tid = len(self.object_id_to_tid) + 1
                    self.object_id_to_tid[ins_id] = tid
                    self.query_feat_tubes.append(
                        SimpleTracker(tid, [None] * frame_id))
                tid = self.object_id_to_tid[ins_id]

                if self.feature_type == 'mask':
                    object_feat = frame_mask_feats[item_idx]
                else:
                    object_feat = _to_numpy_feature(feat)

                qf_tube = self.query_feat_tubes[tid - 1].qf_tube
                missing_frames = frame_id - len(qf_tube)
                if missing_frames > 0:
                    qf_tube.extend([None] * missing_frames)
                qf_tube.append({
                    'query_feat': object_feat,
                    'cls_id': int(ins_id % 1000),
                })

                mask = (pan_results == ins_id).astype(np.uint8)
                mask = mask_utils.encode(np.asfortranarray(mask))
                mask['counts'] = mask['counts'].decode('ascii')
                mask['class_id'] = ins_id % 1000
                online_ids.append(tid)
                online_masks.append(mask)
            self.timer.toc()

        self._write_masks(frame_id + 1, online_masks, online_ids)
        self.frame_id += 1

    def _write_masks(self, frame_id, masks, track_ids):
        for mask, track_id in zip(masks, track_ids):
            if track_id < 0:
                continue
            image_height, image_width = mask['size']
            self.result_file.write(
                '{} {} {} {} {} {}\n'.format(
                    frame_id,
                    track_id,
                    mask['class_id'],
                    image_height,
                    image_width,
                    mask['counts'],
                ))

    def finalize(self):
        self.close()
        for tracker in self.query_feat_tubes:
            missing_frames = self.frame_id - len(tracker.qf_tube)
            if missing_frames > 0:
                tracker.qf_tube.extend([None] * missing_frames)

        feature_filename = (
            'mask_feats.pickle'
            if self.feature_type == 'mask' else 'query_feats.pickle')
        feature_path = osp.join(self.save_root, feature_filename)
        print('Writing results to {}'.format(feature_path), flush=True)
        with open(feature_path, 'wb') as file:
            pickle.dump(self.query_feat_tubes, file)

    def close(self):
        if not self.result_file.closed:
            self.result_file.close()


def concat_seq(outputs, save_root, feature_type='query'):
    writer = VPSSequenceWriter(save_root, feature_type)
    try:
        for output in outputs:
            writer.process(output[0])
    except Exception:
        writer.close()
        raise
    writer.finalize()


# by HB
# not check
def preprocess_video_panoptic_gt(
        gt_labels,
        gt_masks,
        gt_semantic_seg,
        gt_instance_ids,
        num_things,
        num_stuff,
        img_metas,
):
    num_classes = num_things + num_stuff
    num_frames = len(img_metas)

    thing_masks_list = []
    for frame_id in range(num_frames):
        thing_masks_list.append(gt_masks[frame_id].pad(
            img_metas[frame_id]['pad_shape'][:2], pad_val=0).to_tensor(
            dtype=torch.bool, device=gt_labels.device))
    instances = torch.unique(gt_instance_ids[:, 1])
    things_masks = []
    labels = []
    for instance in instances:
        pos_ins = torch.nonzero(torch.eq(gt_instance_ids[:, 1], instance), as_tuple=True)[0]  # 0 is for redundant tuple
        labels_instance = gt_labels[:, 1][pos_ins]
        assert torch.allclose(labels_instance, labels_instance[0])
        labels.append(labels_instance[0])
        instance_frame_ids = gt_instance_ids[:, 0][pos_ins].to(dtype=torch.int32).tolist()
        instance_masks = []
        for frame_id in range(num_frames):
            frame_instance_ids = gt_instance_ids[gt_instance_ids[:, 0] == frame_id, 1]
            if frame_id not in instance_frame_ids:
                empty_mask = torch.zeros(
                    (img_metas[frame_id]['pad_shape'][:2]),
                    dtype=thing_masks_list[frame_id].dtype, device=thing_masks_list[frame_id].device
                )
                instance_masks.append(empty_mask)
            else:
                pos_inner_frame = torch.nonzero(torch.eq(frame_instance_ids, instance), as_tuple=True)[0].item()
                frame_mask = thing_masks_list[frame_id][pos_inner_frame]
                instance_masks.append(frame_mask)
        things_masks.append(torch.stack(instance_masks))

    things_masks = torch.stack(things_masks)
    things_masks = things_masks.to(dtype=torch.long)
    labels = torch.stack(labels)
    labels = labels.to(dtype=torch.long)

    return labels, things_masks


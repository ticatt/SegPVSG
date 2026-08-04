import numpy as np
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms as T

from mmdet.core import INSTANCE_OFFSET


class SingleVideoFrameProcessor:
    """Convert one Mask2Former output into one UniTrack input frame."""

    def __init__(self, data_cfg, tracker_cfg, classes, feature_type='query'):
        data_root = Path(data_cfg.data_root)
        video_name = data_cfg.video_name
        if video_name.startswith('P'):
            data_source = 'epic_kitchen'
        elif video_name.split('_')[0].isdigit() and len(
                video_name.split('_')[0]) == 4:
            data_source = 'vidor'
        else:
            data_source = 'ego4d'

        video_folder = data_root / data_source / 'frames' / video_name
        self.num_classes = len(classes)
        self.feature_type = feature_type
        self.img_files = sorted(str(path) for path in video_folder.rglob('*.png'))
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(tracker_cfg.common.im_mean, tracker_cfg.common.im_std),
        ])

    def __len__(self):
        return len(self.img_files)

    def _get_binary_masks_and_query_feats(self, pan_mask, query_feat_dict):
        object_ids = list(np.unique(pan_mask))
        if self.num_classes in object_ids:
            object_ids.remove(self.num_classes)
        if len(object_ids) == 0:
            return np.array([]), []
        assert len(query_feat_dict) == len(object_ids), (
            'Masks and query feats should match!')

        binary_masks = []
        query_feats = []
        for object_id in object_ids:
            binary_masks.append((pan_mask == object_id).astype(np.int))
            query_feats.append(
                dict(
                    query_feat=self._unify_query_feat_dim(
                        query_feat_dict[object_id]),
                    cls_id=object_id % INSTANCE_OFFSET,
                ))

        return np.stack(binary_masks), query_feats

    @staticmethod
    def _unify_query_feat_dim(query_feat_list):
        if len(query_feat_list) == 1:
            return query_feat_list[0].squeeze()
        query_feat_list = [query_feat.squeeze() for query_feat in query_feat_list]
        return np.stack(query_feat_list).mean(axis=0)

    @staticmethod
    def _pool_mask_features(pan_mask, object_ids, mask_features):
        mask_features_tensor = torch.from_numpy(mask_features).float()
        if mask_features_tensor.ndim == 4:
            mask_features_tensor = mask_features_tensor.squeeze(0)

        masks = np.stack([
            pan_mask == object_id
            for object_id in object_ids
        ]).astype(np.float32)
        masks = torch.from_numpy(masks).unsqueeze(1)
        masks = F.interpolate(
            masks,
            size=mask_features_tensor.shape[-2:],
            mode='nearest',
        ).squeeze(1)

        denom = masks.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        pooled_feats = torch.einsum(
            'nhw,chw->nc', masks, mask_features_tensor) / denom.squeeze(-1)
        return pooled_feats.cpu().numpy()

    def _get_binary_masks_and_mask_feats(
            self, pan_mask, mask_features):
        if mask_features is None:
            raise KeyError(
                'mask_features not found in model output. '
                'Please set feature_type="mask" before inference.')

        object_ids = list(np.unique(pan_mask))
        if self.num_classes in object_ids:
            object_ids.remove(self.num_classes)
        if len(object_ids) == 0:
            return np.array([]), []

        binary_masks = np.stack([
            (pan_mask == object_id).astype(np.int64)
            for object_id in object_ids
        ])
        pooled_feats = self._pool_mask_features(
            pan_mask, object_ids, mask_features)
        mask_feats = [
            dict(query_feat=pooled_feat, cls_id=object_id % INSTANCE_OFFSET)
            for object_id, pooled_feat in zip(object_ids, pooled_feats)
        ]
        return binary_masks, mask_feats

    def process(self, frame_id, frame_output):
        img_ori = cv2.imread(self.img_files[frame_id])
        if img_ori is None:
            raise ValueError('File corrupt {}'.format(self.img_files[frame_id]))

        height, width, _ = img_ori.shape
        img = np.ascontiguousarray((img_ori / 255.)[:, :, ::-1])
        img = self.transforms(img)

        pan_mask = frame_output['pan_results']
        if self.feature_type == 'mask':
            labels, object_feats = self._get_binary_masks_and_mask_feats(
                pan_mask, frame_output.get('mask_features'))
        else:
            labels, object_feats = self._get_binary_masks_and_query_feats(
                pan_mask, frame_output['query_feats'])

        return img, labels, img_ori, (height, width), object_feats


class LoadOutputsFromMask2Former:
    """Legacy sequence adapter retaining the original complete-output API."""

    def __init__(self, data_cfg, outputs, tracker_cfg, classes,
                 feature_type='query'):
        self.frame_processor = SingleVideoFrameProcessor(
            data_cfg=data_cfg,
            tracker_cfg=tracker_cfg,
            classes=classes,
            feature_type=feature_type,
        )
        self.outputs = outputs

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, frame_id):
        return self.frame_processor.process(frame_id, self.outputs[frame_id])

import os
import sys
import pdb
import cv2
import pickle
import yaml
import logging
import argparse
import os.path as osp

import numpy as np
import torch
import pycocotools.mask as mask_utils
from torchvision.transforms import transforms as T

sys.path[0] = os.getcwd()
from models.unitrack.utils.log import logger
from models.unitrack.utils.meter import Timer
from models.unitrack.data.single_video import (
    LoadOutputsFromMask2Former,
    SingleVideoFrameProcessor,
)

from models.unitrack.eval import trackeval
from models.unitrack.eval.mots.MOTSVisualization import MOTSVisualizer
from models.unitrack.utils import visualize as vis
from models.unitrack.utils import io as io
from models.unitrack.mask import MaskAssociationTracker
from models.unitrack.basetrack import BaseTrack


class IPSSequenceWriter:
    """Track one Mask2Former frame at a time and write sequence outputs."""

    def __init__(self, data_cfg, tracker_cfg, classes, save_root,
                 feature_type='query'):
        self.frame_processor = SingleVideoFrameProcessor(
            data_cfg=data_cfg,
            tracker_cfg=tracker_cfg,
            classes=classes,
            feature_type=feature_type,
        )
        self.tracker_cfg = tracker_cfg
        self.feature_type = feature_type
        self.save_root = save_root
        self.frame_id = -1
        self.timer = Timer()

        BaseTrack.reset_count()
        self.tracker = MaskAssociationTracker(tracker_cfg)

        result_filename = osp.join(save_root, 'quantitive', 'masks.txt')
        io.mkdir_if_missing(osp.dirname(result_filename))
        self.result_filename = result_filename
        self.result_file = open(result_filename, 'w')

    @staticmethod
    def _encode_masks(online_targets):
        online_masks = []
        online_ids = []
        for target in online_targets:
            mask = mask_utils.encode(np.asfortranarray(target.mask.astype(np.uint8)))
            mask['counts'] = mask['counts'].decode('ascii')
            mask['class_id'] = target.cls_id
            online_masks.append(mask)
            online_ids.append(target.track_id)
        return online_masks, online_ids

    def _write_frame(self, frame_id, online_masks, online_ids):
        for mask, track_id in zip(online_masks, online_ids):
            if track_id < 0:
                continue
            height, width = mask['size']
            self.result_file.write(
                '{frame} {track_id} {class_id} {height} {width} {counts}\n'.format(
                    frame=frame_id + 1,
                    track_id=track_id,
                    class_id=mask['class_id'],
                    height=height,
                    width=width,
                    counts=mask['counts'],
                ))

    def process(self, frame_output):
        self.frame_id += 1
        if self.frame_id >= len(self.frame_processor):
            raise IndexError('More model outputs than video frames.')

        if self.frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(
                self.frame_id, 1. / max(1e-5, self.timer.average_time)))

        img, obs, img0, _, object_feats = self.frame_processor.process(
            self.frame_id, frame_output)
        if len(obs) == 0:
            self._write_frame(self.frame_id, [], [])
            return

        self.timer.tic()
        online_targets, _ = self.tracker.update(
            img, img0, obs, object_feats, 0)
        online_masks, online_ids = self._encode_masks(online_targets)
        self.timer.toc()
        self._write_frame(self.frame_id, online_masks, online_ids)

    def finalize(self):
        self.result_file.close()
        logger.info('Save results to {}'.format(self.result_filename))

        query_feat_tubes = self.tracker.query_feat_tubes
        query_feat_tubes = [
            query_feat_tube.complete_empty_postfix(self.frame_id)
            for query_feat_tube in query_feat_tubes
        ]
        filename = osp.join(
            self.save_root,
            'mask_feats.pickle' if self.feature_type == 'mask'
            else 'query_feats.pickle',
        )
        print('Writing results to {}'.format(filename), flush=True)
        with open(filename, 'wb') as file:
            pickle.dump(query_feat_tubes, file)
        return query_feat_tubes


def eval_seq(data_cfg,
             tracker_cfg,
             outputs,
             classes,
             save_root,
             return_results=False,
             feature_type='query'):
    save_dir = None
    dataloader = LoadOutputsFromMask2Former(data_cfg=data_cfg,
                                            outputs=outputs,
                                            tracker_cfg=tracker_cfg,
                                            classes=classes,
                                            feature_type=feature_type)
    BaseTrack.reset_count()
    tracker = MaskAssociationTracker(tracker_cfg)
    timer = Timer()
    results = []
    for frame_id, (img, obs, img0, _, object_feats) in enumerate(dataloader):
        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        online_tlwhs = []
        online_ids = []
        online_masks = []
        if len(obs) == 0:
            results.append((frame_id + 1, [], [], []))
        else:
            timer.tic()
            online_targets, _ = tracker.update(img, img0, obs, object_feats, 0)
            for target in online_targets:
                tlwh = target.tlwh * tracker_cfg.common.down_factor
                track_id = target.track_id
                mask = mask_utils.encode(np.asfortranarray(
                    target.mask.astype(np.uint8)))
                mask['counts'] = mask['counts'].decode('ascii')
                mask['class_id'] = target.cls_id
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_masks.append(mask)
            timer.toc()
            results.append(
                (frame_id + 1, online_tlwhs, online_masks, online_ids))
        if save_dir is not None:
            online_im = vis.plot_tracking(img0,
                                          online_masks,
                                          online_ids,
                                          frame_id=frame_id)
            cv2.imwrite(os.path.join(save_dir, '{:04d}.png'.format(frame_id)),
                        online_im)

    result_filename = osp.join(save_root, 'quantitive/masks.txt')
    io.write_mots_results(result_filename, results)

    query_feat_tubes = tracker.query_feat_tubes
    query_feat_tubes = [
        query_feat_tube.complete_empty_postfix(frame_id)
        for query_feat_tube in query_feat_tubes
    ]

    if feature_type == 'mask':
        qf_results_filename = osp.join(save_root, 'mask_feats.pickle')
    else:
        qf_results_filename = osp.join(save_root, 'query_feats.pickle')

    print('Writing results to {}'.format(qf_results_filename), flush=True)
    with open(qf_results_filename, 'wb') as file:
        pickle.dump(query_feat_tubes, file)

    if return_results:
        return results, query_feat_tubes

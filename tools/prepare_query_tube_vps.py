# Copyright (c) OpenMMLab. All rights reserved.
# Copy and modified from mmdet@3b72b12.
import argparse
import gc
import json
import os
import os.path as osp
import warnings

import torch
from mmcv import Config, DictAction, ProgressBar
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                        replace_cfg_vals, setup_multi_processes,
                        update_data_root)

from datasets.datasets.builder import build_dataset
from models.mask2former_vps.utils import VPSSequenceWriter, concat_seq
from utils.gpu import preallocate_cuda_memory


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare VPS tube features')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='directory to save per-video feature files and masks.txt')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--split', help='generate train or val set')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='fuse conv and bn for inference')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='deprecated, use --gpu-id')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use in non-distributed testing')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--eval', type=str, nargs='+')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-dir')
    parser.add_argument('--show-score-thr', type=float, default=0.3)
    parser.add_argument('--gpu-collect', action='store_true')
    parser.add_argument('--tmpdir')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='deprecated, use --eval-options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--feature-type',
        choices=['query', 'mask'],
        default='query',
        help='feature type used for tube feature extraction')
    parser.add_argument('--video-name', default=None, help='only process one video')
    parser.add_argument('--video-start', type=int, default=0, help='start offset for sharded video processing')
    parser.add_argument('--video-step', type=int, default=1, help='step size for sharded video processing')
    parser.add_argument('--skip-existing', action='store_true', help='skip video if target feature file already exists')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot both be specified')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def stream_vps_sequence(model, data_loader, save_root, feature_type):
    model.eval()
    writer = VPSSequenceWriter(save_root, feature_type)
    progress_bar = ProgressBar(len(data_loader.dataset))
    try:
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            frame_output = None
            for frame_output in result:
                writer.process(frame_output[0])
                progress_bar.update()
            del frame_output
            del result
            del data
    except Exception:
        writer.close()
        raise
    writer.finalize()

def main():
    args = parse_args()
    
    if args.eval and args.format_only:
        raise ValueError('--eval and --format-only cannot both be specified')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    if args.work_dir is None:
        raise ValueError('--work-dir is required')
    if args.split is None:
        raise ValueError('--split is required')

    cfg = Config.fromfile(args.config)
    preallocate_cuda_memory()
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('--gpu-ids is deprecated, please use --gpu-id')
    else:
        cfg.gpu_ids = [args.gpu_id]

    cfg.device = get_device()
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=distributed,
        shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

    if hasattr(model, 'module'):
        model.module.feature_type = args.feature_type
    else:
        model.feature_type = args.feature_type

    with open('./data/pvsg.json', 'r') as f:
        anno = json.load(f)

    video_names = []
    for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
        for video_id in anno['split'][data_source][args.split]:
            video_names.append(str(video_id).strip())

    if args.video_name is not None:
        video_names = [args.video_name.strip()]
    else:
        video_names = video_names[args.video_start::args.video_step]

    target_name = 'mask_feats.pickle' if args.feature_type == 'mask' else 'query_feats.pickle'

    for video_name in video_names:
        save_root = osp.join(args.work_dir, video_name)
        target_file = osp.join(save_root, target_name)
        if args.skip_existing and osp.exists(target_file):
            print(f'Skip video {video_name}', flush=True)
            continue

        print(f'Inference for video {video_name}', flush=True)
        cfg.data.test.video_name = video_name
        dataset_single_video = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset_single_video, **test_loader_cfg)
        model.CLASSES = dataset_single_video.CLASSES
        os.makedirs(save_root, exist_ok=True)
        if not distributed and not (args.show or args.show_dir):
            stream_vps_sequence(
                model, data_loader, save_root, args.feature_type)
        else:
            if not distributed:
                outputs = single_gpu_test(
                    model, data_loader, args.show, args.show_dir,
                    args.show_score_thr)
            else:
                outputs = multi_gpu_test(
                    model, data_loader, args.tmpdir,
                    args.gpu_collect or cfg.evaluation.get('gpu_collect', False))
            concat_seq(
                outputs=outputs,
                save_root=save_root,
                feature_type=args.feature_type)
            del outputs

        del data_loader
        del dataset_single_video
        gc.collect()


if __name__ == '__main__':
    main()

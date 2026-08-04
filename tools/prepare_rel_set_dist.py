import argparse
import logging
import multiprocessing
import os
import traceback

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

from tqdm import tqdm

from utils.relation_matching import *


_WORKER_CONTEXT = {}


def parse_args():
    parser = argparse.ArgumentParser(description='prepare relation set')
    parser.add_argument(
        '--data_dir',
        default='./data',
        help='path to pvsg dataset')
    parser.add_argument(
        '--work_dir',
        required=True,
        help='directory containing per-video feature files and masks.txt')
    parser.add_argument(
        '--split',
        required=True,
        choices=['train', 'val'],
        help='generate train or val set')
    parser.add_argument(
        '--feature-type',
        choices=['query', 'mask'],
        default='mask',
        help='which feature pickle to use')
    parser.add_argument(
        '--video-name',
        default=None,
        help='only process one video')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='number of worker processes')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip video if relation pickle already exists')
    parser.add_argument(
        '--gt-time',
        '--gt_time',
        dest='gt_time',
        action='store_true',
        help='use ground-truth temporal spans')
    return parser.parse_args()


def init_worker(context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context.copy()
    _WORKER_CONTEXT['pvsg_dataset'] = PVSGRelationAnnotation(
        f"{_WORKER_CONTEXT['data_dir']}/pvsg.json",
        _WORKER_CONTEXT['split'])


def process_video(vid):
    context = _WORKER_CONTEXT
    pvsg_dataset = context['pvsg_dataset']

    try:
        target_name = (
            'relations_gt_time.pickle'
            if context['gt_time']
            else 'relations.pickle'
        )
        target_file = f"{context['work_dir']}/{vid}/{target_name}"

        if context['skip_existing'] and os.path.exists(target_file):
            return vid, 'skipped', None

        feat_tubes = load_pickle(
            f"{context['work_dir']}/{vid}/{context['feature_file']}")

        pred_mask_tubes = get_pred_mask_tubes_one_video(
            vid,
            context['work_dir'])

        matching_dict = match_and_process_gt_tubes(
            vid,
            pvsg_dataset,
            pred_mask_tubes,
            data_dir=context['data_dir'])

        matching_dict = compact_matching_dict(matching_dict)
        gt_relations = pvsg_dataset[vid]['relations']
        pred_relations = translate_gt_relations(
            matching_dict,
            gt_relations,
            context['gt_time'])

        pred_feat_tubes = {
            feat_tubes[idx].track_id: feat_tubes[idx].qf_tube
            for idx in range(len(feat_tubes))
        }

        if context['gt_time']:
            relation_dict = process_feats_and_relations_gt(
                pred_relations,
                pred_feat_tubes)
        else:
            relation_dict = process_feats_and_relations(
                pred_relations,
                pred_feat_tubes)
            

        save_pickle(target_file, relation_dict)
        return vid, 'completed', None

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = (
            f"\nError processing video {vid}:\n"
            f"- Error Type: {type(e).__name__}\n"
            f"- Error Message: {str(e)}\n"
            f"- Full Traceback:\n{error_trace}"
        )
        return vid, 'error', error_msg


def main():
    args = parse_args()

    pvsg_dataset = PVSGRelationAnnotation(
        f'{args.data_dir}/pvsg.json',
        args.split)

    video_list = [str(vid).strip() for vid in pvsg_dataset.video_ids]
    if args.video_name is not None:
        video_list = [args.video_name.strip()]

    feature_file = (
        'mask_feats.pickle'
        if args.feature_type == 'mask'
        else 'query_feats.pickle'
    )

    context = dict(
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        split=args.split,
        feature_file=feature_file,
        gt_time=args.gt_time,
        skip_existing=args.skip_existing)
    init_worker(context)

    completed = 0
    skipped = 0
    errors = 0

    def report_result(result, pbar=None):
        nonlocal completed, skipped, errors
        vid, status, msg = result

        if status == 'completed':
            completed += 1
        elif status == 'skipped':
            skipped += 1
        else:
            errors += 1

        if pbar is not None:
            pbar.update()
            pbar.set_postfix(
                completed=completed,
                skipped=skipped,
                errors=errors)

        if msg:
            tqdm.write(msg)

        return vid, status

    if args.num_workers <= 1:
        with tqdm(total=len(video_list), desc='Processing videos') as pbar:
            for vid in video_list:
                report_result(process_video(vid), pbar)
    else:
        with multiprocessing.Pool(
                processes=args.num_workers,
                initializer=init_worker,
                initargs=(context,)) as pool:
            with tqdm(total=len(video_list), desc='Processing videos') as pbar:
                for result in pool.imap_unordered(
                        process_video, video_list, chunksize=1):
                    report_result(result, pbar)


if __name__ == '__main__':
    main()

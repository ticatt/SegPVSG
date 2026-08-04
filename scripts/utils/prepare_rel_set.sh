#!/usr/bin/env bash
# Build relation-set files from prepared IPS or VPS per-video features.


usage() {
  cat <<'EOF_HELP'
Usage: bash scripts/utils/prepare_rel_set.sh [OPTIONS]

Options:
  --data_dir DIR             PVSG data directory. Default: ./data
  --split {train,val}        Default: train
  --work_dir DIR             Default: outputs/features/ips_train_vf
  --feature-type {mask,query} Default: mask
  --num-workers N            Worker processes. Default: 4
  --video-name ID            Process one video only.
  --gt-time, --gt_time       Use ground-truth temporal spans.
  -h, --help                 Show this help message.

Existing relation files are skipped. Other options are forwarded unchanged to
 tools/prepare_rel_set_dist.py.
EOF_HELP
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

cd "${PROJECT_ROOT}"

python tools/prepare_rel_set_dist.py \
    --data_dir ./data \
    --split train \
    --feature-type mask \
    --work_dir outputs/features/ips_train_vf \
    --num-workers 4 \
    --skip-existing \
    "$@"

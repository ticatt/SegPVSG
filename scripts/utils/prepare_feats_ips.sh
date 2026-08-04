#!/usr/bin/env bash
# Extract IPS tube features and forward remaining arguments to the Python entrypoint.


usage() {
  cat <<'EOF_HELP'
Usage: bash scripts/utils/prepare_feats_ips.sh [SCRIPT OPTIONS] [FEATURE OPTIONS]

Script options:
  --gpu ID           CUDA GPU ID. Default: 0
  --cpu-threads N    OMP and MKL thread count. Default: 16
  -h, --help         Show this help message.

Common feature options:
  --split {train,val}        Default: train
  --work-dir DIR             Default: outputs/features/ips_train_vf
  --feature-type {mask,query} Default: mask
  --video-name ID            Process one video only.
  --video-start N            Shard start offset. Default: 0
  --video-step N             Shard stride. Default: 1

Existing completed videos are skipped. Other feature options are forwarded
unchanged to tools/prepare_query_tube_ips.py.
EOF_HELP
}
GPU=0
CPU_THREADS=16
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --gpu=*)
      GPU="${1#*=}"
      shift
      ;;
    --cpu-threads)
      CPU_THREADS="$2"
      shift 2
      ;;
    --cpu-threads=*)
      CPU_THREADS="${1#*=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"

export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}ignore:__floordiv__ is deprecated:UserWarning,ignore:torch.meshgrid:UserWarning"

cd "${PROJECT_ROOT}"

python tools/prepare_query_tube_ips.py \
  configs/unitrack/imagenet_resnet50_s3_womotion_timecycle.py \
  checkpoints/mask2former_r50_ips/epoch_8.pth \
  --work-dir outputs/features/ips_train_vf \
  --split train \
  --feature-type mask \
  --launcher none \
  --skip-existing \
  "${ARGS[@]}"

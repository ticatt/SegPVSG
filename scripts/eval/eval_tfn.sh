#!/usr/bin/env bash
# Evaluate a TFN checkpoint, loading the experiment's saved YAML config by default.


usage() {
  cat <<'EOF_HELP'
Usage: bash scripts/eval/eval_tfn.sh [OPTIONS]

Options:
  --gpu ID          CUDA GPU ID. Default: 0
  --exp-name NAME   Experiment name under --save-root.
  --config PATH     Config override. Otherwise loads the experiment's config.yaml.
  --save-root DIR   Experiment root. Default: outputs/results
  --save-dir DIR    Direct experiment directory override.
  --checkpoint PATH Checkpoint path override.
  --epoch NAME      Checkpoint suffix, for example epoch_8 or epoch_best.
  --eval-tag NAME   Optional suffix for evaluation result files.
  -h, --help        Show this help message.
EOF_HELP
}
GPU=0
CONFIG=
EXP_NAME=
SAVE_ROOT=outputs/results
SAVE_DIR=
CHECKPOINT=
EPOCH=
EVAL_TAG=

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
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --epoch)
      EPOCH="$2"
      shift 2
      ;;
    --eval-tag)
      EVAL_TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="$GPU"

CMD=(python tools/run_eval.py --save-root "$SAVE_ROOT")
if [[ -n "$CONFIG" ]]; then
  CMD+=(--config "$CONFIG")
fi
if [[ -n "$EXP_NAME" ]]; then
  CMD+=(--exp-name "$EXP_NAME")
fi
if [[ -n "$SAVE_DIR" ]]; then
  CMD+=(--save-dir "$SAVE_DIR")
fi
if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$EPOCH" ]]; then
  CMD+=(--epoch "$EPOCH")
fi
if [[ -n "$EVAL_TAG" ]]; then
  CMD+=(--eval-tag "$EVAL_TAG")
fi

"${CMD[@]}"

#!/usr/bin/env bash
# Train TFN from a YAML configuration and save run artifacts under outputs/results.


usage() {
  cat <<'EOF_HELP'
Usage: bash scripts/train/train_tfn.sh [OPTIONS]

Options:
  --gpu ID          CUDA GPU ID. Default: 0
  --config PATH     Training YAML. Default: configs/tfn/train_ips.yaml
  --exp-name NAME   Override the experiment name from the YAML file.
  --save-root DIR   Override train.save_root from the YAML file.
  -h, --help        Show this help message.
EOF_HELP
}
GPU=0
CONFIG=configs/tfn/train_ips.yaml
EXP_NAME=
SAVE_ROOT=

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
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="$GPU"

CMD=(python tools/run_train.py --config "$CONFIG")
if [[ -n "$EXP_NAME" ]]; then
  CMD+=(--exp-name "$EXP_NAME")
fi
if [[ -n "$SAVE_ROOT" ]]; then
  CMD+=(--save-root "$SAVE_ROOT")
fi

"${CMD[@]}"

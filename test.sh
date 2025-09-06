#!/usr/bin/env bash

usage() {
  cat <<USAGE
Usage: $0 -d DATASET [-r ROUNDS] [-g GPU_ID] [-o OUTPUT_PATH]

Available DATASET options (default rounds):
  WebNLG       (rounds=4)
  WebNLG_star  (rounds=2)
  NYT24        (rounds=3)
  NYT24_star   (rounds=2)
  NYT29        (rounds=3)
USAGE
}

# default parameters
 dataset="WebNLG"
 rounds=4
 gpu_id=0
 output_path=./ckpt

# flags to track if dataset/rounds provided
 dataset_cli=0
 rounds_cli=0

while getopts "d:r:g:o:h" opt; do
    case ${opt} in
      d) dataset=${OPTARG}; dataset_cli=1 ;;
      r) rounds=${OPTARG}; rounds_cli=1 ;;
      g) gpu_id=${OPTARG} ;;
      o) output_path=${OPTARG} ;;
      h) usage; exit 0 ;;
      *) usage; exit 1 ;;
    esac
done

# set default rounds based on dataset
if [ ${dataset_cli} -eq 1 ] && [ ${rounds_cli} -eq 0 ]; then
  case ${dataset} in
    WebNLG) rounds=4 ;;
    WebNLG_star) rounds=2 ;;
    NYT24) rounds=3 ;;
    NYT24_star) rounds=2 ;;
    NYT29) rounds=3 ;;
    *) echo "Unknown dataset: ${dataset}" >&2; usage; exit 1 ;;
  esac
else
  case ${dataset} in
    WebNLG|WebNLG_star|NYT24|NYT24_star|NYT29) ;;
    *) echo "Unknown dataset: ${dataset}" >&2; usage; exit 1 ;;
  esac
fi

ckpt_dir="${output_path}/${dataset}"
if [ ! -f "${ckpt_dir}/best_model.bin" ]; then
  echo "Model not found in ${ckpt_dir}. Please train first." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpu_id} python -u run.py \
  --cuda_id=${gpu_id} \
  --dataset=${dataset} \
  --train=test \
  --rounds=${rounds} \
  --output_path="${output_path}"

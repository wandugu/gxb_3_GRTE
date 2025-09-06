#!/usr/bin/env bash

usage() {
  cat <<USAGE
Usage: $0 -d DATASET [-r ROUNDS] [-e EPOCHS] [-g GPU_ID] [-o OUTPUT_PATH] [-s SAVE_INTERVAL]

Available DATASET options (default rounds):
  WebNLG       (rounds=4)
  WebNLG_star  (rounds=2)
  NYT24        (rounds=3)
  NYT24_star   (rounds=2)
  NYT29        (rounds=3)
USAGE
}

# default parameters，可在此处直接修改
dataset="WebNLG"
rounds=4
epochs=50
gpu_id=1
output_path=./ckpt
save_interval=1

# 标记是否通过参数传入 dataset 或 rounds
dataset_cli=0
rounds_cli=0

while getopts "d:r:e:g:o:s:h" opt; do
    case ${opt} in
      d) dataset=${OPTARG}; dataset_cli=1 ;;
      r) rounds=${OPTARG}; rounds_cli=1 ;;
      e) epochs=${OPTARG} ;;
      g) gpu_id=${OPTARG} ;;
      o) output_path=${OPTARG} ;;
      s) save_interval=${OPTARG} ;;
      h) usage; exit 0 ;;
      *) usage; exit 1 ;;
    esac
done

# 根据传入的数据集设置默认轮数
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
load_model=""
if [ -f "${ckpt_dir}/best_model.bin" ]; then
  load_model="--load_model"
fi

CUDA_VISIBLE_DEVICES=${gpu_id} python -u run.py \
  --cuda_id=${gpu_id} \
  --dataset=${dataset} \
  --train=train \
  --rounds=${rounds} \
  --num_train_epochs=${epochs} \
  --output_path="${output_path}" \
  --save_interval=${save_interval} \
  ${load_model}


#!/bin/bash

# run eval with the pretrained model
python ./experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /hpc2hdd/home/haoangli/wenxuan/model_checkpoint/openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

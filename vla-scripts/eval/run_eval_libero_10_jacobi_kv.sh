#!/bin/bash



# module load cuda/11.8

# run eval with the pretrained model

CUDA_VISIBLE_DEVICES=0 python ./experiments/robot/libero/run_libero_eval_jacobi_kv.py\
  --model_family openvla \
  --pretrained_checkpoint /hpc2hdd/home/haoangli/wenxuan/ckpt/openvla-7b+libero_10_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug-ac3-80000-ckpt \
  --task_suite_name libero_10 \
  --center_crop True \
  --action_chunk 3 \

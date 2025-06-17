#!/bin/bash

module load cuda/11.8
# source ~/user68/conda_env/openvla/bin/activate
# export PYTHONPATH=/data/user/wsong890/user68/project/openvla-consitent:$PYTHONPATH
# export PYTHONPATH=/data/user/wsong890/user68/project/LIBERO:$PYTHONPATH
# 设置环境变量
export DATA_ROOT_DIR="/hpc2hdd/home/haoangli/wenxuan/project/openvla-consitent/modified_libero_rlds/"  #libero_10_no_noops path
export RUN_ROOT_DIR="./run/libero_10_ac3" # save train result path                 
export ADAPTER_TMP_DIR="./run/libero_10_ac3/adapter_weights"       

# 创建必要的目录
mkdir -p $RUN_ROOT_DIR
mkdir -p $ADAPTER_TMP_DIR
# export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=0,1

# lora fitune
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "/hpc2hdd/home/haoangli/wenxuan/project/openvla-consitent/model_ckpt/openvla-7b" \
  --data_root_dir $DATA_ROOT_DIR \
  --dataset_name libero_10_no_noops \
  --run_root_dir $RUN_ROOT_DIR \
  --adapter_tmp_dir $ADAPTER_TMP_DIR \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 10000 \
  --wandb_project openvla-finetune-ac3 \
  --window_size 1 \
  --future_action_window_size 2 \
  --max_steps 100000 \
  --save_latest_checkpoint_only False \
  --num_images_in_input 1


# CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding

<a href="https://arxiv.org/abs/2506.13725" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/CEED--VLA-red?label=arXiv&color=red
" height="25" />
</a>
<a href="https://irpn-eai.github.io/CEED-VLA/" target="_blank">
    <img alt="project" src="https://img.shields.io/badge/CEED--VLA-blue?label=Project&color=blue" height="25" />
</a>
<a href="https://huggingface.co/robovlms/RoboVLMs" target="_blank">
    <img alt="HF Model: CEED-VLA" src="https://img.shields.io/badge/CEED--VLA-yellow?label=Model(no consistnent traning)&color=ffd400" height="25" />
</a>
<br>

**Wenxuan Song¹\***, **Jiayi Chen¹\***, **Pengxiang Ding²˒³†**, **Yuxin Huang¹**, **Han Zhao²˒³**, **Donglin Wang²**,  
**Haoang Li¹‡**

¹ IRPN Lab, HKUST(GZ) <br>
² MiLab, Westlake University  <br> 
³ Zhejiang University  

\* Equal Contribution  † Project Leader  ‡ Corresponding Author



 
[**Overview**](#overview) | [**Installation**](#installation) | [**Get Checkpoint**](#get-the-checkpoint) | [**Evaluation**](#libero-evaluations) | [**Results**](#results-of-openvla-on-libero-long-speed-in-tokenss) 


<hr style="border: 2px solid gray;"></hr>

## Overview

We now only open-source the **consistent-training-free version** of CEED-VLA base on OpenVLA.  
The consistent training code , jacobi trajectory generating code and LLaVA-VLA version will be release later.


## Installation

**Note:** If you already have the [OpenVLA](https://github.com/openvla/openvla) environment installed, you can skip this section.


### 1. Install OpenVLA.

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/IRPN-EAI/CEED-VLA.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 2. Install Libero

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

Download dataset
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```




## Get the checkpoint
### Download checkpoint
We provide a pretrained checkpoint of OpenVLA with **action chunk = 3**.

```bash

git lfs install

git clone https://huggingface.co/chenpyyy/openvla-ac
```

### (option) Train a checkpoint with action chunk=3 by youself

**Note:** 
Before running the finetuning script, please download [OpenVLA](https://huggingface.co/openvla/openvla-7b) from Hugging Face and place it in ./model_ckpt/openvla-7b.
Then, replace the following files in the downloaded model with those in our repo:
```bash

Replace:
  ./model_ckpt/openvla-7b/configuration_prismatic.py
  ./model_ckpt/openvla-7b/modeling_prismatic.py
  ./model_ckpt/openvla-7b/processing_prismatic.py

With:
  ./prismatic/extern/hf/configuration_prismatic.py
  ./prismatic/extern/hf/modeling_prismatic.py
  ./prismatic/extern/hf/processing_prismatic.py

```
train!
```bash
#!/bin/bash

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
  --vla_path "./model_ckpt/openvla-7b" \
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

```


## LIBERO Evaluations

### Results of OpenVLA in LIBERO-LONG Environment on H100(Speed in tokens/s)



| Decoding Method | Action Chunk | Success Rate| Min Speed | Max Speed | Avg. Speed |  
|--------|----------------|---------------|-------------|-------------|---------|
| AR | 1 | 53.0  | 48.1  | 55.3  | 51.0  |
| AR | 3| 60.4  | 49.3  | 60.4  | 54.3  |
| Jacobi (ours) | 3| **62.4**  | **62.9**  | **163.1** | **85.0** |



Each success rate is the 500 rollouts each (10 tasks x 50 rollouts per task).

### Run Evaluation Script

```bash
bash ./vla-scripts/eval/run_eval_libero_10_jacobi_kv.sh

```
run_eval_libero_10_jacobi_kv.sh
```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./experiments/robot/libero/run_libero_eval_jacobi_kv.py\
  --model_family openvla \
  --pretrained_checkpoint ./ckpt/openvla-7b+libero_10_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug-ac3-80000-ckpt \
  --task_suite_name libero_10 \
  --center_crop True \
  --action_chunk 3 \

```


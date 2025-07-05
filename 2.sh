#!/bin/bash
#BSUB -m gpu07                    # 明确调度空闲节点
#BSUB -gpu "num=1:mode=exclusive_process"  # 分配1张GPU卡
#BSUB -R "rusage[mem=32G]"             # 分配32GB内存（按你需要的大小设置）
#BSUB -W 04:00                         # 最多运行时间4小时
#BSUB -o logs/vllm_output.%J.log       # 标准输出日志
#BSUB -e logs/vllm_error.%J.err        # 错误日志

# 创建日志目录（如果没提前建好）
mkdir -p logs

nvidia-smi

export CUDA_HOME=/nfsshare/apps/cuda-11.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64

python -m 2.py \
--model ./dg/iic/cv_dla34_table-structure-recognition_cycle-centernet

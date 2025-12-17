#!/bin/bash
#SBATCH -J eval_mae
#SBATCH -p gpu                  # 按你集群改：gpu / a100 / compute 等
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# 无显示器环境避免 matplotlib 报错（即使脚本里没用也无害）
export MPLBACKEND=Agg

# 线程数（可选）
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ===== 激活你的环境（按实际改）=====
       # 你的 conda 环境名若不同就改这里

# ===== 运行 =====
/home/qygao/.conda/envs/matten/bin/python -u /home/qygao/matten/scripts/eval_test_mae.py \
  --config /home/qygao/matten/scripts/configs/materials_tensor_dielectric.yaml \
  --ckpt /home/qygao/matten/scripts/wandb_logs/matten_dielectric_tensor/s97m3vmq/checkpoints/epoch=55-step=20384.ckpt \
  --out /home/qygao/matten/datasets/dielectric_tensor_distribution/test_mae_ten.json \
  --device cuda



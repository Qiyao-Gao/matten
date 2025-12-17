#!/bin/bash -l
#SBATCH --job-name=matten_test          # 作业名称
#SBATCH --output=slurm_di_log/slurm-%j.out           # 输出文件(%j替换为作业ID)
#SBATCH --error=slurm_di_log/slurm-%j.err            # error文件（%j替换为作业ID)                                         
#SBATCH --partition=gpu                 # 使用计算分区 
#SBATCH --nodes=1                       # 节点数
#SBATCH --ntasks-per-node=1             # 每个节点任务数
#SBATCH --cpus-per-task=1              # 每个节点任务数
#SBATCH --time=02-00:00:00              # 运行时间(天-时:分:秒)
#SBATCH --gres=gpu:1                    # ✅ 申请1张GPU

conda activate matten
export WANDB_BASE_URL="https://api.bandw.top"
#python train_materials_tensor.py        # 执行计算任务
#python train_materials_tensor_piezoelectric.py
python train_materials_tensor_dielectric.py
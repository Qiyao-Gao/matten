#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Local runner for Matten (WSL/Linux)
# Usage:
#   bash scripts/run_di_local.sh                # default: dielectric
#   bash scripts/run_di_local.sh piezo          # run piezoelectric
#   bash scripts/run_di_local.sh scalar         # run train_materials_tensor.py
# Env:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_di_local.sh
# -------------------------

MODE="${1:-dielectric}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- pick which python entry ----
case "${MODE}" in
  dielectric|di)
    PY_ENTRY="./train_materials_tensor_dielectric/train_materials_tensor_dielectric.py"
    ;;
  piezo|piezoelectric)
    PY_ENTRY="./train_materials_tensor_dielectric/train_materials_tensor_piezoelectric.py"
    ;;
  scalar|base)
    PY_ENTRY="./train_materials_tensor_dielectric/train_materials_tensor.py"
    ;;
  *)
    echo "[ERROR] Unknown MODE: ${MODE}"
    echo "        Supported: dielectric|di, piezo|piezoelectric, scalar|base"
    exit 1
    ;;
esac

# ---- conda activation ----
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "[ERROR] conda not found. Please open a shell where conda works or edit this script."
    exit 1
  fi
fi
conda activate matten

# ---- your env var ----
export WANDB_BASE_URL="https://api.bandw.top"

# ---- logs ----
LOG_DIR="slurm_di_log"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_LOG="${LOG_DIR}/local-${MODE}-${TS}.out"
ERR_LOG="${LOG_DIR}/local-${MODE}-${TS}.err"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Mode: ${MODE}"
echo "[INFO] Entry: ${PY_ENTRY}"
echo "[INFO] WANDB_BASE_URL=${WANDB_BASE_URL}"
if [ "${CUDA_VISIBLE_DEVICES:-}" != "" ]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] stdout -> ${OUT_LOG}"
echo "[INFO] stderr -> ${ERR_LOG}"
echo

python "${PY_ENTRY}" \
  1> >(tee "${OUT_LOG}") \
  2> >(tee "${ERR_LOG}" >&2)

echo
echo "[INFO] Done."

# How to run:
# cd /mnt/d/学习/科研/文明健/matten
# bash scripts/run_di_local.sh           # 默认跑 dielectric
# bash scripts/run_di_local.sh piezo     # 跑 piezo
# CUDA_VISIBLE_DEVICES=0 bash scripts/run_di_local.sh dielectric

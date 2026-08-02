bash <<'BASH'
set -Eeuo pipefail

# ============================================================
# I-JEPA / LeJEPA optimizer sweep on 8 GPUs
#
# Runs, sequentially:
#   1. I-JEPA  + AdamW baseline
#   2. I-JEPA  + Muon
#   3. I-JEPA  + Stiefel Riemannian Adam
#   4. LeJEPA  + AdamW baseline
#   5. LeJEPA  + Muon
#   6. LeJEPA  + Stiefel Riemannian Adam
# ============================================================

REPO="$(pwd)"
DATA_ROOT="/ssd/zhangjie/imagenet100/imagefolder"
TRAIN_DIR="${DATA_ROOT}/train"
VAL_DIR="${DATA_ROOT}/val"
RUN_ROOT="/ssd/zhangjie/ssl_runs/jepa_optimizer_sweep"
LOG_ROOT="${RUN_ROOT}/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "[0/4] Repository and environment check"
echo "repo: ${REPO}"
echo "============================================================"

test -f "${REPO}/main_pretrain.py"
test -d "${REPO}/solo"
test -d "${REPO}/scripts/pretrain/imagenet-100"

# python -m pip install -e .

python - <<'PY'
import hydra
import lightning
import torch
import torchvision

print(f"torch       : {torch.__version__}")
print(f"torchvision : {torchvision.__version__}")
print(f"lightning   : {lightning.__version__}")
print(f"hydra-core  : {hydra.__version__}")
print(f"CUDA ready  : {torch.cuda.is_available()}")
print(f"GPU count   : {torch.cuda.device_count()}")

assert torch.cuda.is_available(), "CUDA is unavailable."
assert torch.cuda.device_count() >= 8, "At least 8 GPUs are required."
PY

echo
echo "============================================================"
echo "[1/4] GPU check"
echo "============================================================"

nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader

GPU_COUNT="$(nvidia-smi -L | wc -l)"
if [ "${GPU_COUNT}" -lt 8 ]; then
  echo "ERROR: detected ${GPU_COUNT} GPU(s), but this sweep requires at least 8."
  exit 1
fi

echo
echo "============================================================"
echo "[2/4] ImageNet-100 ImageFolder check"
echo "============================================================"

test -d "${TRAIN_DIR}"
test -d "${VAL_DIR}"

TRAIN_CLASSES="$(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
VAL_CLASSES="$(find "${VAL_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)"

echo "train classes: ${TRAIN_CLASSES}"
echo "val classes  : ${VAL_CLASSES}"

if [ "${TRAIN_CLASSES}" -ne 100 ] || [ "${VAL_CLASSES}" -ne 100 ]; then
  echo "ERROR: expected 100 class folders in both train and val."
  exit 1
fi

if ! diff \
  <(find "${TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
  <(find "${VAL_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort); then
  echo "ERROR: train and val class directory names do not match."
  exit 1
fi

python - <<PY
from torchvision.datasets import ImageFolder

train = ImageFolder("${TRAIN_DIR}")
val = ImageFolder("${VAL_DIR}")

print(f"train samples: {len(train)}")
print(f"val samples  : {len(val)}")
print(f"classes      : {len(train.classes)}")
print(f"first classes: {train.classes[:5]}")

assert len(train.classes) == 100
assert train.classes == val.classes
assert len(train) > 0
assert len(val) > 0
PY

echo
echo "============================================================"
echo "[3/4] Hydra configuration check"
echo "============================================================"

python main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name ijepa.yaml \
  devices=8 \
  ++strategy=ddp_find_unused_parameters_true \
  precision=bf16-mixed \
  wandb.enabled=true \
  --cfg job \
  > "${LOG_ROOT}/ijepa_config_check.txt"

python main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name lejepa.yaml \
  devices=8 \
  ++strategy=ddp_find_unused_parameters_true \
  precision=bf16-mixed \
  wandb.enabled=true \
  --cfg job \
  > "${LOG_ROOT}/lejepa_config_check.txt"

echo "Configuration checks passed."

echo
echo "============================================================"
echo "[4/4] Start full optimizer sweep"
echo "Logs: ${LOG_ROOT}"
echo "============================================================"

run_experiment () {
  local run_name="$1"
  shift

  echo
  echo "------------------------------------------------------------"
  echo "START: ${run_name}"
  echo "TIME : $(date '+%F %T')"
  echo "------------------------------------------------------------"

  torchrun --standalone --nproc_per_node=8 \
    main_pretrain.py \
    "$@" \
    devices=8 \
    ++strategy=ddp_find_unused_parameters_true \
    precision=bf16-mixed \
    data.num_workers=8 \
    wandb.enabled=true \
    name="${run_name}" \
    checkpoint.frequency=1 \
    auto_resume.enabled=true \
    2>&1 | tee "${LOG_ROOT}/${run_name}.log"

  echo "------------------------------------------------------------"
  echo "DONE : ${run_name}"
  echo "TIME : $(date '+%F %T')"
  echo "------------------------------------------------------------"
}

# ------------------------------------------------------------
# 1) I-JEPA — original AdamW baseline
# YAML base:
#   AdamW, lr=1e-3, betas=(0.9, 0.95), wd=0.04, 300 epochs
# ------------------------------------------------------------
run_experiment "ijepa_adamw_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name ijepa.yaml \
  checkpoint.dir="${RUN_ROOT}/ijepa_adamw"

# ------------------------------------------------------------
# 2) I-JEPA — Muon
#
# At global batch 1024, optimizer.lr=0.005 is linearly scaled
# to an effective Muon hidden-matrix LR of 0.02.
#
# aux_lr is NOT batch-size-scaled:
# it is the AdamW LR for patch embedding, predictor, norm,
# bias, classifier and other non-Muon parameters.
# ------------------------------------------------------------
run_experiment "ijepa_muon_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name ijepa.yaml \
  optimizer.name=muon \
  optimizer.lr=0.005 \
  ++optimizer.kwargs.aux_lr=0.001 \
  ++optimizer.kwargs.momentum=0.95 \
  ++optimizer.kwargs.ns_steps=5 \
  ++optimizer.kwargs.nesterov=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/ijepa_muon"

# ------------------------------------------------------------
# 3) I-JEPA — Stiefel Riemannian Adam
# Same I-JEPA LR / scheduler / weight decay as AdamW baseline.
# Only encoder hidden matrix weights use tangent projection
# and QR retraction; heads and auxiliary parameters use AdamW.
# ------------------------------------------------------------
run_experiment "ijepa_riemannian_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name ijepa.yaml \
  optimizer.name=riemannian \
  ++optimizer.kwargs.manifold=stiefel \
  ++optimizer.kwargs.retraction=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/ijepa_riemannian"

# ------------------------------------------------------------
# 4) LeJEPA — original AdamW baseline
# YAML base:
#   AdamW, lr=5e-4, betas=(0.9, 0.999), wd=1e-4, 400 epochs
# ------------------------------------------------------------
run_experiment "lejepa_adamw_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name lejepa.yaml \
  checkpoint.dir="${RUN_ROOT}/lejepa_adamw"

# ------------------------------------------------------------
# 5) LeJEPA — Muon
#
# Muon hidden-matrix effective LR: 0.02.
# AdamW fallback LR for ResNet input conv, projector, norm,
# bias and classifier: 5e-4, matching the LeJEPA baseline.
# ------------------------------------------------------------
run_experiment "lejepa_muon_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name lejepa.yaml \
  optimizer.name=muon \
  optimizer.lr=0.005 \
  ++optimizer.kwargs.aux_lr=0.0005 \
  ++optimizer.kwargs.momentum=0.95 \
  ++optimizer.kwargs.ns_steps=5 \
  ++optimizer.kwargs.nesterov=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/lejepa_muon"

# ------------------------------------------------------------
# 6) LeJEPA — Stiefel Riemannian Adam
# Same LeJEPA LR / scheduler / weight decay as AdamW baseline.
# ------------------------------------------------------------
run_experiment "lejepa_riemannian_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name lejepa.yaml \
  optimizer.name=riemannian \
  ++optimizer.kwargs.manifold=stiefel \
  ++optimizer.kwargs.retraction=true \
  ++optimizer.kwargs.betas=[0.9,0.999] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/lejepa_riemannian"

echo
echo "============================================================"
echo "ALL SIX EXPERIMENTS FINISHED SUCCESSFULLY"
echo "Finish time: $(date '+%F %T')"
echo "Logs       : ${LOG_ROOT}"
echo "Checkpoints: ${RUN_ROOT}"
echo "============================================================"
BASH
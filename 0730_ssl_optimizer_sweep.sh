#!/usr/bin/env bash
set -Eeuo pipefail

# BYOL / DINO / MAE optimizer study on ImageNet-100.
# Each method retains its repository YAML (architecture, augmentation,
# schedule, epoch budget, and original baseline optimizer).  Muon and
# Riemannian variants override only optimizer-specific fields.

REPO="/home/zhangjie/solo-learn"
RUN_ROOT="/ssd/zhangjie/ssl_runs/ssl_optimizer_sweep_0730"
LOG_ROOT="${RUN_ROOT}/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${LOG_ROOT}"
cd "${REPO}"

FAILED_RUNS=()

run_experiment() {
  local run_name="$1"
  shift

  echo "============================================================"
  echo "START: ${run_name}"
  echo "TIME : $(date '+%F %T')"
  echo "============================================================"

  if torchrun --standalone --nproc_per_node=8 \
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
    2>&1 | tee "${LOG_ROOT}/${run_name}.log"; then
    echo "DONE : ${run_name}"
  else
    echo "FAILED: ${run_name}"
    FAILED_RUNS+=("${run_name}")
  fi
  echo "TIME : $(date '+%F %T')"
}

# Original ImageNet-100 baselines (repository YAMLs):
# BYOL/DINO use LARS; MAE uses AdamW. All YAMLs specify 400 epochs.
run_experiment "byol_lars_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name byol.yaml \
  checkpoint.dir="${RUN_ROOT}/byol_lars"

# Muon: matrix LR 0.005 is linearly scaled to 0.02 at global batch 1024.
# The AdamW-style auxiliary LR is deliberately not batch-scaled.
run_experiment "byol_muon_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name byol.yaml \
  optimizer.name=muon \
  optimizer.lr=0.005 \
  ++optimizer.kwargs.aux_lr=0.001 \
  ++optimizer.kwargs.momentum=0.95 \
  ++optimizer.kwargs.ns_steps=5 \
  ++optimizer.kwargs.nesterov=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/byol_muon"

# Riemannian ResNet setting: base LR 5e-4 -> effective LR 2e-3 at batch 1024.
run_experiment "byol_riemannian_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name byol.yaml \
  optimizer.name=riemannian \
  optimizer.lr=0.0005 \
  ++optimizer.kwargs.manifold=stiefel \
  ++optimizer.kwargs.retraction=true \
  ++optimizer.kwargs.betas=[0.9,0.999] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/byol_riemannian"

run_experiment "dino_lars_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name dino.yaml \
  checkpoint.dir="${RUN_ROOT}/dino_lars"

run_experiment "dino_muon_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name dino.yaml \
  optimizer.name=muon \
  optimizer.lr=0.005 \
  ++optimizer.kwargs.aux_lr=0.001 \
  ++optimizer.kwargs.momentum=0.95 \
  ++optimizer.kwargs.ns_steps=5 \
  ++optimizer.kwargs.nesterov=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/dino_muon"

run_experiment "dino_riemannian_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name dino.yaml \
  optimizer.name=riemannian \
  optimizer.lr=0.0005 \
  ++optimizer.kwargs.manifold=stiefel \
  ++optimizer.kwargs.retraction=true \
  ++optimizer.kwargs.betas=[0.9,0.999] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/dino_riemannian"

run_experiment "mae_adamw_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name mae.yaml \
  checkpoint.dir="${RUN_ROOT}/mae_adamw"

run_experiment "mae_muon_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name mae.yaml \
  optimizer.name=muon \
  optimizer.lr=0.005 \
  ++optimizer.kwargs.aux_lr=0.0002 \
  ++optimizer.kwargs.momentum=0.95 \
  ++optimizer.kwargs.ns_steps=5 \
  ++optimizer.kwargs.nesterov=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/mae_muon"

# Conservative ViT Riemannian setting: base LR 5e-5 -> effective LR 2e-4.
run_experiment "mae_riemannian_8x5090" \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name mae.yaml \
  optimizer.name=riemannian \
  optimizer.lr=0.00005 \
  ++optimizer.kwargs.manifold=stiefel \
  ++optimizer.kwargs.retraction=true \
  ++optimizer.kwargs.betas=[0.9,0.95] \
  ++optimizer.kwargs.eps=1e-8 \
  checkpoint.dir="${RUN_ROOT}/mae_riemannian"

echo "============================================================"
if [ "${#FAILED_RUNS[@]}" -eq 0 ]; then
  echo "ALL NINE EXPERIMENTS FINISHED SUCCESSFULLY"
else
  echo "COMPLETED WITH FAILURES: ${FAILED_RUNS[*]}"
fi
echo "Finish time: $(date '+%F %T')"
echo "Logs: ${LOG_ROOT}"
echo "Checkpoints: ${RUN_ROOT}"
echo "============================================================"

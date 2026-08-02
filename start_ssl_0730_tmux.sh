#!/usr/bin/env bash
# Minimal non-interactive tmux launcher: use the already-created environment
# directly, so this does not depend on a login shell loading conda functions.
set -Eeuo pipefail

export PATH="/home/zhangjie/miniconda3/envs/solo-learn/bin:/home/zhangjie/miniconda3/bin:${PATH}"
cd /home/zhangjie/solo-learn
exec bash ./0730_ssl_optimizer_sweep.sh

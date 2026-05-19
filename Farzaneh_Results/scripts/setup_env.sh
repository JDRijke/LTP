#!/usr/bin/env bash
# One-time environment setup on Hábrók (RUG).
# Adjust module names to match the cluster's current toolchain.

set -euo pipefail

module purge
module load Python/3.11.5-GCCcore-13.2.0     # check `module avail Python` for current versions
module load CUDA/12.1.1                       # match the driver on the A100 nodes

ENV_DIR=${ENV_DIR:-$HOME/envs/touche2026}
mkdir -p "$(dirname "$ENV_DIR")"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
# Torch wheel that matches the CUDA module above:
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -r requirements.txt

echo "Env ready at $ENV_DIR"
echo "Activate with:  source $ENV_DIR/bin/activate"

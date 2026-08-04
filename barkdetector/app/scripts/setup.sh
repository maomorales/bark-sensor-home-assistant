#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VENV_DIR="${PROJECT_ROOT}/.venv"

echo "==> Creating Python virtual environment in ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip and installing dependencies"
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "==> Creating runtime directories (may require sudo)"
sudo mkdir -p /var/log/barkdetector
sudo mkdir -p /var/lib/barkdetector/captures
sudo chown "$(whoami)" /var/log/barkdetector
sudo chown "$(whoami)" /var/lib/barkdetector/captures

echo "==> Setup complete."
echo "Edit ${PROJECT_ROOT}/config/config.yaml to match your environment."
echo "List microphones via: ${PROJECT_ROOT}/main.py --list-devices"
echo "Run detector: ${PROJECT_ROOT}/main.py --config ${PROJECT_ROOT}/config/config.yaml"

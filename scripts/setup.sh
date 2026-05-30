#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Crop Recommendation System — initial workspace setup (Linux / macOS / Git Bash)
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "==> Project root: ${PROJECT_ROOT}"

# 1. Create virtual environment
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
else
  echo "==> Virtual environment already exists at ${VENV_DIR}"
fi

# 2. Activate and install dependencies
echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing requirements"
pip install -r "${PROJECT_ROOT}/requirements.txt"

echo ""
echo "Setup complete. Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. Place raw CSV in data/raw/crop_recommendation.csv"
echo "  3. python src/train.py"
echo "  4. streamlit run app/app.py"

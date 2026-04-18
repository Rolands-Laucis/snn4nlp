#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

echo "Activating..."
source "$VENV_DIR/bin/activate"

echo "Verifying install from req.txt..."
pip install -r req.txt

echo "Done. Environment ready."
#!/usr/bin/env bash
# Reproduce every result from scratch (macOS / Linux / Git Bash). ~25-30 min on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -m venv .venv
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m oncogene.cli all          # download -> preprocess -> analyze -> train -> figures -> report
python -m pytest                    # unit + data + dashboard tests
python scripts/build_notebooks.py --execute
echo "Done. Launch the dashboard with:  streamlit run dashboard/app.py"

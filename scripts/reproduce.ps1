# Reproduce every result from scratch (Windows PowerShell). ~25-30 min on a laptop CPU.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m oncogene.cli all
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python scripts\build_notebooks.py --execute
Write-Host "Done. Launch the dashboard with:  .\.venv\Scripts\streamlit run dashboard\app.py"

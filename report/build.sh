#!/bin/sh
# Regenerate figures/tables from the code, then compile the PDF.
#   sh report/build.sh            (from the project root)
set -e
cd "$(dirname "$0")/.."
.venv/bin/python scripts/make_report_assets.py
cd report && "${TECTONIC:-$HOME/.local/bin/tectonic}" -X compile main.tex
echo "report/main.pdf"

#!/bin/sh
set -eu
cd "$(dirname "$0")"
export DYLD_LIBRARY_PATH="$PWD/work/runtime${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export MPLCONFIGDIR="$PWD/work/matplotlib"
export IPYTHONDIR="$PWD/work/ipython"
.venv/bin/python -m pytest -q
.venv/bin/python main.py --stage all --offline --end 2026-09-01T00:00:00Z

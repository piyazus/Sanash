#!/usr/bin/env bash
# Shared helpers: resolve the python and kaggle entry points.
# Source this, don't run it: . "$(dirname "$0")/_env.sh"
#
# On Windows, pip installs kaggle.exe into a per-user Scripts dir that is
# usually not on PATH, and `python3` resolves to the Microsoft Store stub.
# Both fall back to forms that work everywhere.

if command -v python &> /dev/null && python -c "" &> /dev/null; then
  PY=python
elif command -v python3 &> /dev/null && python3 -c "" &> /dev/null; then
  PY=python3
else
  echo "No working python on PATH."
  exit 1
fi

if command -v kaggle &> /dev/null; then
  KAGGLE=(kaggle)
elif "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('kaggle') else 1)"; then
  # kaggle has no __main__, so the module form targets kaggle.cli directly.
  KAGGLE=("$PY" -m kaggle.cli)
else
  echo "kaggle CLI not found. Install with: pip install kaggle --break-system-packages"
  exit 1
fi

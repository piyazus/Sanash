#!/usr/bin/env bash
# Verify kaggle CLI is installed and credentials are in place.
# Get kaggle.json from https://www.kaggle.com/settings -> API -> Create New Token
# then place it yourself at ~/.kaggle/kaggle.json (chmod 600).
# Claude Code should never handle the token contents directly.
set -euo pipefail
. "$(dirname "$0")/_env.sh"

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Missing ~/.kaggle/kaggle.json. Download it from kaggle.com/settings"
  echo "and place it manually — do not paste the token contents into chat."
  exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"
"${KAGGLE[@]}" datasets list -s "test" > /dev/null && echo "Kaggle auth OK"

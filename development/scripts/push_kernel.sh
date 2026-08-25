#!/usr/bin/env bash
# Push a notebook/script as a Kaggle kernel and start execution.
# Usage: bash push_kernel.sh <path-to-kernel-dir-with-kernel-metadata.json>
#
# Each kernel dir needs a kernel-metadata.json (kaggle kernels init -p <dir>
# generates a template). Set enable_gpu: true only when you actually need it —
# it burns weekly GPU quota (30h/week free tier).
set -euo pipefail
. "$(dirname "$0")/_env.sh"

KERNEL_DIR="${1:?Usage: push_kernel.sh <kernel-dir>}"

echo "Pushing kernel from $KERNEL_DIR ..."
"${KAGGLE[@]}" kernels push -p "$KERNEL_DIR"

SLUG=$("$PY" -c "import json; print(json.load(open('$KERNEL_DIR/kernel-metadata.json'))['id'])")
echo "Pushed as $SLUG. Poll status with:"
echo "  ${KAGGLE[*]} kernels status $SLUG"
echo "Pull output once complete with:"
echo "  ${KAGGLE[*]} kernels output $SLUG -p ./outputs"

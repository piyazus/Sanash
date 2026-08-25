#!/usr/bin/env bash
# Resumable download of the Gorelik et al. multi-view in-cabin dataset (73.5GB).
# DOI: 10.5281/zenodo.20559664
set -euo pipefail

DEST_DIR="${1:-./data/raw}"
URL="https://zenodo.org/records/20559664/files/beintelli_v1.zip?download=1"
EXPECTED_MD5="74924b5be59e706a5a89affba48f6b87"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Downloading to $DEST_DIR/beintelli_v1.zip (resumable, 73.5GB — this will take a while)"
# Git Bash ships curl but not wget, so prefer whichever exists.
if command -v wget &> /dev/null; then
  wget -c "$URL" -O beintelli_v1.zip
elif command -v curl &> /dev/null; then
  curl -L -C - "$URL" -o beintelli_v1.zip
else
  echo "Neither wget nor curl found."
  exit 1
fi

echo "Verifying checksum..."
ACTUAL_MD5=$(md5sum beintelli_v1.zip | awk '{print $1}')
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
  echo "MD5 mismatch! Expected $EXPECTED_MD5, got $ACTUAL_MD5. Re-run to resume/retry."
  exit 1
fi
echo "Checksum OK."

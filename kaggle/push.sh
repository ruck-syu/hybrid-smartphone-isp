#!/bin/bash
# usage: kaggle/push.sh <slug> <model> "<extra train args>" [extra dataset sources comma-sep]
set -e
cd "$(dirname "$0")/.."
SLUG=$1; MODEL=$2; EXTRA=$3; SRC=${4:-}
D=kaggle/kernels/$SLUG; mkdir -p $D
sed -e "s|__MODEL__|$MODEL|" -e "s|__EXTRA__|$EXTRA|" kaggle/kernel_run.py > $D/run.py
DS='"dedruck/isp-code", "skibidiohiorizz/sidd-medium-raw", "rajat95gupta/smartphone-image-denoising-dataset"'
[ -n "$SRC" ] && DS="$DS, $SRC"
cat > $D/kernel-metadata.json <<J
{"id": "dedruck/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_internet": true, "machine_shape": "NvidiaTeslaT4",
 "dataset_sources": [$DS], "competition_sources": [], "kernel_sources": []}
J
kaggle kernels push -p $D

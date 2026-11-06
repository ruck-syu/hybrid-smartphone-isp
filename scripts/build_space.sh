#!/bin/bash
# Assemble the deploy bundle (Railway Docker build; also HF-Space compatible) in build/space (ONNX-only, no torch).
set -e
cd "$(dirname "$0")/.."
rm -rf build/space && mkdir -p build/space/models build/space/eval
cp -r isp app samples build/space/
rm -rf build/space/isp/__pycache__ build/space/app/__pycache__
cp models/*.onnx build/space/models/ 2>/dev/null || true
cp eval/RESULTS.md build/space/eval/ 2>/dev/null || true
cp requirements.txt build/space/requirements.txt
cp Dockerfile build/space/Dockerfile
printf "__pycache__\n*.pyc\n" > build/space/.dockerignore
cp docs/architecture/hybrid-isp.png build/space/ 2>/dev/null || true
cat > build/space/README.md <<'MD'
---
title: Hybrid Smartphone ISP
emoji: 📷
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: GRADIO_VERSION
app_file: app/app.py
pinned: false
license: mit
short_description: Hybrid classical + learned RAW-to-sRGB smartphone ISP
---
Hybrid learned smartphone ISP (classical WB / CCM / gamma + noise-conditioned INT8 U-Net).
Source, docs and evaluation: see the GitHub repository linked in the app.
MD
GV=$(.venv/bin/python -c "import gradio;print(gradio.__version__)")
sed -i '' "s/GRADIO_VERSION/$GV/" build/space/README.md
du -sh build/space

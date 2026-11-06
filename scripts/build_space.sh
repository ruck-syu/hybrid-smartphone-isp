#!/bin/bash
# Assemble the deploy bundle for the Railway Docker build in build/space (ONNX-only, no torch).
set -e
cd "$(dirname "$0")/.."
rm -rf build/space && mkdir -p build/space/models build/space/eval
cp -r isp app samples build/space/
find build/space -name __pycache__ -type d -prune -exec rm -rf {} +
cp models/*.onnx build/space/models/ 2>/dev/null || true
cp eval/*.py eval/RESULTS.md build/space/eval/ 2>/dev/null || true
mkdir -p build/space/eval/results && cp eval/results/*.json build/space/eval/results/ 2>/dev/null || true
cp requirements.txt build/space/requirements.txt
cp Dockerfile build/space/Dockerfile
printf "__pycache__\n*.pyc\n" > build/space/.dockerignore
cp docs/architecture/hybrid-isp.png build/space/ 2>/dev/null || true
cat > build/space/README.md <<'MD'
Deploy bundle of the Hybrid Smartphone ISP web app (built by scripts/build_space.sh).
Run: python app/server.py
MD
du -sh build/space

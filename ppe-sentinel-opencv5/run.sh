#!/bin/bash
cd "$(dirname "$0")"
[ -f ppe_model.onnx ] || { echo "Put ppe_model.onnx in this folder first (see README)."; exit 1; }
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt
echo "Open http://127.0.0.1:8000"
uvicorn app:app --reload

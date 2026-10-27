# ppe-sentinel-opencv5
# PPE//SENTINEL

Minimal PPE compliance monitor for the OpenCV AI Competition 2026. It watches a camera or video file, flags missing safety gear (helmet, vest and more), and auto-saves every violation (head and face areas blurred) so logs can be reviewed and downloaded as CSV or ZIP.

- Demo video: YOUR_YOUTUBE_LINK
- Live guided tour (simulated data, nothing to install): YOUR_VERCEL_LINK
- Competition: https://opencv26.devpost.com

## Run it
1. Download `ppe_model.onnx` (v2 folder) from https://huggingface.co/ayushgupta7777/safetyvision-yolov8 (YOLOv8, AGPL-3.0).
2. Put it inside the `ppe-sentinel-opencv5` folder of this repo.
3. `cd ppe-sentinel-opencv5 && bash run.sh`
4. Open http://127.0.0.1:8000 and press **Try demo**.

## Use it
1. Pick a source (camera or video file).
2. Choose the gear you require.
3. Press **Start monitoring**. Violations are saved to `logs/` automatically.
4. Download logs as CSV or a ZIP of frames.

Also: sensitivity slider, pop-up and sound alerts, motion meter (OpenCV frame differencing), optional AWS S3 + SNS sync (set `S3_BUCKET` and `SNS_TOPIC_ARN`).

## Built with
OpenCV 5 (DNN inference, motion detection, blur), SafetyVision YOLOv8 (ayushgupta7777, AGPL-3.0), ONNX, FastAPI, vanilla JS/CSS. AI-assisted (Claude, Gemini, Arena AI) and human-tested.

## Honest limits
- Prototype. I did not train the model, and accuracy depends on it.
- Saved frames blur head and face regions; the live preview on your own screen is not blurred.
- The demo tour uses simulated data and saves nothing.
- A person should always review alerts.

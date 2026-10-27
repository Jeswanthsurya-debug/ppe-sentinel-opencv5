# PPE//SENTINEL

Minimal PPE compliance monitor built for the OpenCV AI Competition 2026. It watches a camera or video file, flags missing safety gear, and saves every violation automatically (with head and face regions blurred) so it can be reviewed and downloaded later.

## Links
- Demo video: YOUR_YOUTUBE_LINK
- Live guided tour (simulated data, nothing to install): YOUR_VERCEL_LINK
- Competition: https://opencv26.devpost.com

## Run it
1. Download the model `ppe_model.onnx` from https://huggingface.co/ayushgupta7777/safetyvision-yolov8 (YOLOv8, AGPL-3.0, v2 folder) and put it in this folder.
2. `bash run.sh`
3. Open http://127.0.0.1:8000 and press **Try demo** for a 30-second tour.

## Use it
1. Pick a source (camera or video file).
2. Choose the gear you require (Hardhat, Safety Vest, Goggles, Mask, Gloves, optional Fall alert).
3. Press **Start monitoring**. Violations are saved to `logs/` automatically.
4. Download the logs any time as CSV or a ZIP of frames.

Other controls: sensitivity slider, pop-up and sound alerts, motion meter (OpenCV frame differencing), optional "pause when scene is still".

## Optional AWS sync
```
export S3_BUCKET=your-bucket
export SNS_TOPIC_ARN=arn:aws:sns:...   # optional
bash run.sh
```
Needs AWS credentials (`aws configure`) and `boto3` (already in requirements). Without these the app runs local-only and says so in the header badge.

## Honest limits
- Prototype. Accuracy depends on the detection model, which was not trained or evaluated by this project.
- Saved violation frames blur the head area of detected people and face-related boxes (OpenCV 5 no longer ships Haar cascades). The live preview on your own screen is not blurred.
- The "Try demo" tour (also hosted from `web-demo/`) uses simulated data and saves nothing. Real detection needs the local backend.
- A person should always review alerts.

## Credits
Model: SafetyVision YOLOv8 by ayushgupta7777 (AGPL-3.0). Built with OpenCV 5, FastAPI and ONNX.

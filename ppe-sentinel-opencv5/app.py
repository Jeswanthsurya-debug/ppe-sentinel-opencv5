import ast, csv, io, json, os, threading, time, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, onnx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

MODEL = os.getenv("MODEL_PATH", "ppe_model.onnx")
CONF, NMS, COOLDOWN = 0.35, 0.45, 10
S3_BUCKET, SNS_ARN = os.getenv("S3_BUCKET"), os.getenv("SNS_TOPIC_ARN")
LOGS = Path("logs"); LOGS.mkdir(exist_ok=True)

meta = onnx.load(MODEL, load_external_data=False)
props = {p.key: p.value for p in meta.metadata_props}
NAMES = ast.literal_eval(props["names"]) if "names" in props else {}
dims = meta.graph.input[0].type.tensor_type.shape.dim
SZ = (dims[2].dim_value if len(dims) > 2 else 0) or 640
net = cv2.dnn.readNetFromONNX(MODEL)
print("classes:", NAMES or "NO NAMES IN MODEL METADATA", "| input size:", SZ)

status = {"cloud": bool(S3_BUCKET), "uploaded": 0, "last": "", "error": "", "cv": cv2.__version__}
last_seen = {}

def is_viol(n): return n.upper().startswith("NO")

def detect(img, thr=CONF):
    h, w = img.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(img, 1 / 255, (SZ, SZ), swapRB=True))
    out = net.forward()[0]
    if out.shape[0] < out.shape[1]: out = out.T
    sc = out[:, 4:]; cid = sc.argmax(1); conf = sc.max(1)
    k = conf > thr
    out, cid, conf = out[k], cid[k], conf[k]
    if len(out) == 0: return []
    boxes = np.stack([(out[:, 0] - out[:, 2] / 2) * w / SZ, (out[:, 1] - out[:, 3] / 2) * h / SZ,
                      out[:, 2] * w / SZ, out[:, 3] * h / SZ], 1)
    keep = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), thr, NMS)).flatten()
    res = []
    for i in keep:
        n = NAMES.get(int(cid[i]), f"class_{cid[i]}")
        res.append({"box": [round(float(v)) for v in boxes[i]], "name": n,
                    "conf": round(float(conf[i]), 2), "violation": is_viol(n)})
    return res

def cloud(eid, ev):
    if not S3_BUCKET: return
    try:
        import boto3
        s3, day = boto3.client("s3"), ev["time"][:10]
        for ext in ("jpg", "json"):
            s3.upload_file(str(LOGS / f"{eid}.{ext}"), S3_BUCKET, f"violations/{day}/{eid}.{ext}")
        if SNS_ARN:
            boto3.client("sns").publish(TopicArn=SNS_ARN, Subject="PPE violation",
                Message=f"{', '.join(ev['types'])} on {ev['camera']} at {ev['time']}")
        status.update(uploaded=status["uploaded"] + 1, last=ev["time"], error="")
    except Exception as e:
        status["error"] = str(e)[:200]

def save(img, dets, cam, up=True):
    now = time.time()
    types = [t for t in sorted({d["name"] for d in dets if d["violation"]}) if now - last_seen.get(t, 0) > COOLDOWN]
    if not types: return
    for t in types: last_seen[t] = now
    vis = img.copy()
    H, W = vis.shape[:2]
    def blur(x, y, w, h):
        x, y = max(0, x), max(0, y); x2, y2 = min(W, x + w), min(H, y + h)
        if x2 > x and y2 > y: vis[y:y2, x:x2] = cv2.GaussianBlur(vis[y:y2, x:x2], (61, 61), 0)
    for d in dets:  # OpenCV 5 has no Haar cascades: blur head of each person + face-related boxes
        x, y, w, h = d["box"]; n = d["name"].upper()
        if n == "PERSON": blur(x, y, w, int(h * 0.28))
        elif any(k in n for k in ("GOGGLES", "MASK", "HARDHAT")):
            blur(x - int(w * .5), y - int(h * .5), int(w * 2), int(h * 2))
    for d in dets:
        x, y, w, h = d["box"]; c = (108, 93, 255) if d["violation"] else (212, 234, 94)
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
        cv2.putText(vis, d["name"], (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    eid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:5]
    cv2.imwrite(str(LOGS / f"{eid}.jpg"), vis)
    ev = {"id": eid, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "camera": cam,
          "types": types, "confidence": max(d["conf"] for d in dets if d["violation"])}
    (LOGS / f"{eid}.json").write_text(json.dumps(ev))
    if up: cloud(eid, ev)

app = FastAPI()
app.mount("/logs", StaticFiles(directory="logs"), name="logs")

@app.get("/")
def index(): return FileResponse("index.html")

@app.get("/intro.mp4")
def intro(): return FileResponse("intro.mp4")

@app.get("/status")
def get_status(): return status

@app.get("/events")
def events():
    return [json.loads(f.read_text()) for f in sorted(LOGS.glob("*.json"), reverse=True)[:50]]

ITEMS = (("HARDHAT", "Hardhat"), ("VEST", "Safety Vest"), ("GOGGLES", "Goggles"), ("MASK", "Mask"), ("GLOVES", "Gloves"), ("FALL", "Fall alert"))
def item(n):
    u = n.upper()
    return next((lbl for key, lbl in ITEMS if key in u), None)

prev = {"g": None}
def motion_pct(img):
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    p, prev["g"] = prev["g"], g
    if p is None: return 100.0
    return round(100 * float((cv2.absdiff(g, p) > 25).mean()), 1)

@app.get("/export.csv")
def export_csv():
    b = io.StringIO(); w = csv.writer(b); w.writerow(["time", "camera", "violations", "confidence", "frame"])
    for f in sorted(LOGS.glob("*.json")):
        e = json.loads(f.read_text()); w.writerow([e["time"], e["camera"], "; ".join(e["types"]), e["confidence"], e["id"] + ".jpg"])
    return Response(b.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=ppe_violations.csv"})

@app.get("/export.zip")
def export_zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for f in LOGS.iterdir(): z.write(f, f.name)
    return Response(b.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=ppe_logs.zip"})

@app.post("/detect_frame")
async def detect_frame(file: UploadFile = File(...), camera: str = Form("cam-1"), required: str = Form(""),
                       conf: float = Form(CONF), cloud_on: int = Form(1), gate: int = Form(0)):
    t0 = time.time()
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return JSONResponse({"error": "bad frame"}, 400)
    mo = motion_pct(img)
    if gate and mo < 1.0:
        return {"skipped": True, "motion": mo, "ms": round((time.time() - t0) * 1000)}
    dets = detect(img, min(max(conf, 0.1), 0.9))
    req = {r for r in required.split(",") if r}
    if "Fall alert" not in req: dets = [d for d in dets if item(d["name"]) != "Fall alert"]
    for d in dets:
        d["item"] = item(d["name"])
        d["violation"] = (is_viol(d["name"]) and (not req or d["item"] in req)) or d["item"] == "Fall alert"
    viol = [d for d in dets if d["violation"]]
    worn = [d for d in dets if not d["violation"] and d["name"].lower() != "person"]
    score = round(100 * len(worn) / (len(worn) + len(viol))) if worn or viol else 100
    if viol: threading.Thread(target=save, args=(img, dets, camera, bool(cloud_on)), daemon=True).start()
    return {"boxes": dets, "violations": sorted({d["name"] for d in viol}), "score": score, "motion": mo,
            "workers": sum(d["name"].lower() == "person" for d in dets),
            "ms": round((time.time() - t0) * 1000), "w": img.shape[1], "h": img.shape[0]}

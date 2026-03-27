"""
CivicNode AI Server
-------------------
1. Ambil daftar kamera aktif dari backend (GET /api/cctv/active)
2. Buka stream DroidCam per kamera di background thread
3. Jalankan YOLO inference, gambar bounding box ke frame
4. Expose MJPEG stream per kamera di /stream/<cctv_id>
5. Kirim hasil deteksi ke backend (POST /api/detection) tiap DETECTION_INTERVAL detik
"""

import cv2
import time
import threading
import requests
from datetime import datetime, timezone
from flask import Flask, Response
from ultralytics import YOLO

from config import (
    BACKEND_URL, AI_SERVER_SECRET,
    MODEL_PATH, CONFIDENCE_THRESHOLD,
    STREAM_PORT, DETECTION_INTERVAL,
)
from utils import get_device
from camera import VideoStream

app = Flask(__name__)

# State per kamera: { cctv_id: { zona_id, frame (bytes|None), detections, lock } }
camera_states: dict = {}


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def fetch_active_cameras() -> list:
    """Ambil daftar kamera aktif dari backend."""
    try:
        res = requests.get(
            f"{BACKEND_URL}/api/cctv/active",
            headers={"x-ai-secret": AI_SERVER_SECRET},
            timeout=10,
        )
        res.raise_for_status()
        return res.json().get("data", [])
    except Exception as e:
        print(f"[server] Gagal ambil daftar kamera: {e}")
        return []


# ---------------------------------------------------------------------------
# Per-camera YOLO thread
# ---------------------------------------------------------------------------

def process_camera(cctv_id: str, stream_url: str, model: YOLO, device: str):
    """Baca frame dari kamera, jalankan YOLO, simpan hasil ke camera_states."""
    print(f"[{cctv_id[:8]}] Koneksi ke {stream_url} ...")
    cap = VideoStream(src=stream_url)
    time.sleep(1)  # tunggu buffer pertama

    state = camera_states[cctv_id]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        results = model.predict(frame, device=device, conf=CONFIDENCE_THRESHOLD, verbose=False)
        annotated = results[0].plot()

        # Kumpulkan deteksi dari semua bounding box
        detections = []
        for box in results[0].boxes:
            cls_id     = int(box.cls[0])
            label      = model.names[cls_id]
            confidence = round(float(box.conf[0]), 4)
            detections.append({"jenis_objek": label, "confidence": confidence})

        # Encode frame ke JPEG (kualitas 80 — cukup untuk streaming)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with state["lock"]:
            state["frame"]      = buffer.tobytes()
            state["detections"] = detections


# ---------------------------------------------------------------------------
# Detection sender thread
# ---------------------------------------------------------------------------

def send_detections_loop():
    """Kirim deteksi semua kamera ke backend tiap DETECTION_INTERVAL detik."""
    while True:
        time.sleep(DETECTION_INTERVAL)
        waktu = datetime.now(timezone.utc).isoformat()

        for cctv_id, state in camera_states.items():
            with state["lock"]:
                detections = list(state["detections"])

            payload = {
                "cctv_id"   : cctv_id,
                "zona_id"   : state["zona_id"],
                "detections": detections,
                "waktu"     : waktu,
            }

            try:
                requests.post(
                    f"{BACKEND_URL}/api/detection",
                    json=payload,
                    headers={"x-ai-secret": AI_SERVER_SECRET},
                    timeout=5,
                )
            except Exception as e:
                print(f"[detection] Gagal kirim {cctv_id[:8]}: {e}")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

def generate_mjpeg(cctv_id: str):
    """Generator frame MJPEG untuk satu kamera."""
    state = camera_states.get(cctv_id)
    if not state:
        return

    while True:
        with state["lock"]:
            frame = state["frame"]

        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(1 / 30)  # ~30fps


@app.route("/stream/<cctv_id>")
def stream(cctv_id):
    if cctv_id not in camera_states:
        return {"error": "Kamera tidak ditemukan"}, 404
    return Response(
        generate_mjpeg(cctv_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "cameras": [
            {"cctv_id": cid, "nama": state.get("nama", ""), "active": state["frame"] is not None}
            for cid, state in camera_states.items()
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=== CivicNode AI Server ===")

    # 1. Fetch kamera dari backend
    print("[server] Mengambil daftar kamera dari backend...")
    cameras = fetch_active_cameras()
    if not cameras:
        print("[server] Tidak ada kamera aktif. Pastikan backend jalan dan ada kamera dengan status=true.")
        return

    print(f"[server] {len(cameras)} kamera ditemukan.")

    # 2. Load YOLO model — satu instance, dipakai semua thread
    print("[server] Loading YOLO model...")
    model  = YOLO(MODEL_PATH)
    device = get_device()

    # 3. Init state + start thread per kamera
    for cam in cameras:
        cid = cam["id"]
        camera_states[cid] = {
            "nama"      : cam.get("nama", cid[:8]),
            "zona_id"   : cam["zona_id"],
            "frame"     : None,
            "detections": [],
            "lock"      : threading.Lock(),
        }
        t = threading.Thread(
            target=process_camera,
            args=(cid, cam["stream_url"], model, device),
            daemon=True,
        )
        t.start()
        print(f"[server] Thread kamera '{cam.get('nama', cid[:8])}' dimulai.")

    # 4. Start detection sender thread
    t = threading.Thread(target=send_detections_loop, daemon=True)
    t.start()
    print(f"[server] Detection sender aktif, interval {DETECTION_INTERVAL} detik.")

    # 5. Start Flask
    print(f"\n[server] Server jalan di http://0.0.0.0:{STREAM_PORT}")
    print(f"[server] Stream URL: http://<IP_SERVER>:{STREAM_PORT}/stream/<cctv_id>")
    print(f"[server] Health:     http://<IP_SERVER>:{STREAM_PORT}/health\n")
    app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True)


if __name__ == "__main__":
    main()

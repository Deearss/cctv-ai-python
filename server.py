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
from ultralytics import YOLO, RTDETR

from config import (
    BACKEND_URL, AI_SERVER_SECRET,
    MODEL_PATH, CONFIDENCE_THRESHOLD,
    STREAM_PORT, DETECTION_INTERVAL,
)
from utils import get_device
from camera import VideoStream
from layer1_tracker import StationaryTracker

app = Flask(__name__)

# State per kamera: { cctv_id: { zona_id, frame (bytes|None), detections, lock } }
camera_states: dict = {}

# ---------------------------------------------------------------------------
# Mapping kelas COCO → kategori sampah
# Kelas yang tidak ada di sini dianggap bukan sampah dan tidak dikirim ke backend
# ---------------------------------------------------------------------------
TRASH_MAP: dict[str, str] = {
    # Plastik
    "bottle"      : "botol plastik",
    "cup"         : "gelas plastik",
    "bowl"        : "wadah plastik",
    "toothbrush"  : "sampah plastik",
    # Sisa makanan
    "banana"      : "sisa makanan",
    "apple"       : "sisa makanan",
    "orange"      : "sisa makanan",
    "sandwich"    : "sisa makanan",
    "hot dog"     : "sisa makanan",
    "pizza"       : "sisa makanan",
    "donut"       : "sisa makanan",
    "cake"        : "sisa makanan",
    "carrot"      : "sisa makanan",
    "broccoli"    : "sisa makanan",
    # Peralatan makan sekali pakai
    "fork"        : "peralatan makan sekali pakai",
    "knife"       : "peralatan makan sekali pakai",
    "spoon"       : "peralatan makan sekali pakai",
    # Kaca
    "wine glass"  : "gelas kaca",
    "vase"        : "pecahan kaca",
    # Elektronik
    "cell phone"  : "sampah elektronik",
    "remote"      : "sampah elektronik",
    "keyboard"    : "sampah elektronik",
    "laptop"      : "sampah elektronik",
    "tv"          : "sampah elektronik",
    "hair drier"  : "sampah elektronik",
    # Lain-lain
    "umbrella"    : "sampah umum",
    "handbag"     : "sampah umum",
    "backpack"    : "sampah umum",
    "suitcase"    : "sampah umum",
    "tie"         : "sampah umum",
    "book"        : "sampah kertas",
    "scissors"    : "sampah tajam",
    "teddy bear"  : "sampah umum",
    "frisbee"     : "sampah umum",
    "sports ball" : "sampah umum",
    "kite"        : "sampah umum",
}


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

def process_camera(cctv_id: str, stream_url: str, model, device: str):
    """Baca frame dari kamera, jalankan model, track dengan IoU, simpan hasil ke camera_states."""
    print(f"[{cctv_id[:8]}] Koneksi ke {stream_url} ...")
    cap = VideoStream(src=stream_url)
    time.sleep(1)  # tunggu buffer pertama

    last_predict_time = 0
    cached_boxes = []  # format: [(x1, y1, x2, y2, color, label_text), ...]
    PREDICT_INTERVAL = DETECTION_INTERVAL  # Pake interval dari config (misal 60 detik)
    
    # Init Tracker buat ngecek sampah yang udah diem 5 menit (300 detik)
    tracker = StationaryTracker(iou_threshold=0.45, max_missed=2, abandon_time_seconds=300)
    state = camera_states[cctv_id]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        current_time = time.time()
        annotated = frame.copy()

        # 1. Jalankan prediksi AI (Layer 2) hanya jika interval terjadwal terpenuhi
        if current_time - last_predict_time >= PREDICT_INTERVAL:
            results = model.predict(frame, device=device, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=320)
            
            raw_detections = []
            
            for box in results[0].boxes:
                cls_id     = int(box.cls[0])
                coco_label = model.names[cls_id]
                jenis      = TRASH_MAP.get(coco_label) or coco_label # Gunakan label default (car, person, dll) jika bukan sampah
                confidence = round(float(box.conf[0]), 4)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                raw_detections.append((x1, y1, x2, y2, jenis, confidence))

            # Teruskan ke Layer 1 (Math Tracker)
            tracked_objects = tracker.update(raw_detections)
            new_boxes = []
            backend_valid_detections = []

            for obj in tracked_objects:
                x1, y1, x2, y2 = obj.box
                tracked_time = int(current_time - obj.first_seen)
                
                if obj.is_abandoned:
                    color = (0, 0, 255) # MERAH (Valid Abandoned)
                else:
                    color = (0, 255, 255) # KUNING (Sedang dilacak)
                    
                label_text = f"{obj.jenis} {obj.confidence} ({tracked_time}s)"
                new_boxes.append((x1, y1, x2, y2, color, label_text))
                
                # Kirim ke backend supaya dihitung
                backend_valid_detections.append({
                    "jenis_objek": obj.jenis, 
                    "confidence": obj.confidence,
                    "status": "abandoned" if obj.is_abandoned else "tracking"
                })

            cached_boxes = new_boxes
            last_predict_time = current_time

            # Simpan deteksi untuk backend thread
            with state["lock"]:
                state["detections"] = backend_valid_detections

        # 2. Gambar kotak bounding box dari hasil cache terbaru untuk SEMUA frame (sangat cepat)
        for x1, y1, x2, y2, color, label_text in cached_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label_text, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Encode frame ke JPEG (kualitas 60 cukup, kurangi bandwidth mjpeg)
        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        _, raw_buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])

        with state["lock"]:
            state["frame"] = buffer.tobytes()
            state["raw_frame"] = raw_buffer.tobytes()

        # Biar CPU nggak mentok 100% nge-loop terus karena cap.read() instan
        time.sleep(0.03)


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
# Global Instances
# ---------------------------------------------------------------------------
global_model = None
global_device = None

# ... (rest of imports and initialization)

def sync_cameras():
    """Ambil daftar kamera aktif terbaru dan spawn thread untuk kamera baru."""
    global global_model, global_device
    if global_model is None or global_device is None:
        return

    cameras = fetch_active_cameras()
    if not cameras:
        return

    for cam in cameras:
        cid = cam["id"]
        if cid not in camera_states:
            camera_states[cid] = {
                "nama"      : cam.get("nama", cid[:8]),
                "zona_id"   : cam["zona_id"],
                "frame"     : None,
                "raw_frame" : None,
                "detections": [],
                "lock"      : threading.Lock(),
            }
            t = threading.Thread(
                target=process_camera,
                args=(cid, cam["stream_url"], global_model, global_device),
                daemon=True,
            )
            t.start()
            print(f"[server] Thread kamera (dinamis) '{cam.get('nama', cid[:8])}' dimulai.")

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

def generate_mjpeg(cctv_id: str, raw: bool = False):
    """Generator frame MJPEG untuk satu kamera."""
    state = camera_states.get(cctv_id)
    if not state:
        return

    while True:
        with state["lock"]:
            frame = state["raw_frame"] if raw else state["frame"]

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
        # Kamera belum dimuat, coba sinkronisasi ulang
        sync_cameras()
        if cctv_id not in camera_states:
            return {"error": "Kamera tidak ditemukan"}, 404
        # Tunggu sebentar agar thread baru sempat membaca frame pertama
        time.sleep(1.5)

    return Response(
        generate_mjpeg(cctv_id, raw=False),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/raw_stream/<cctv_id>")
def raw_stream(cctv_id):
    if cctv_id not in camera_states:
        sync_cameras()
        if cctv_id not in camera_states:
            return {"error": "Kamera tidak ditemukan"}, 404
        time.sleep(1.5)

    return Response(
        generate_mjpeg(cctv_id, raw=True),
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
    global global_model, global_device
    print("=== CivicNode AI Server ===")

    # 1. Load Model (RT-DETR atau YOLO biasa)
    print(f"[server] Loading Model ({MODEL_PATH})...")
    if "rtdetr" in MODEL_PATH.lower():
        global_model = RTDETR(MODEL_PATH)
    else:
        global_model = YOLO(MODEL_PATH)
    global_device = get_device()

    # 2. Fetch kamera dari backend dan spawn thread utama
    print("[server] Mengambil daftar kamera dari backend...")
    sync_cameras()
    
    if not camera_states:
        print("[server] Tidak ada kamera aktif saat startup. Server tetap berjalan dan akan memuat kamera secara dinamis.")

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

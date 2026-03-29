"""
CivicNode AI Server (Headless Data Node)
-------------------
1. Ambil daftar kamera aktif dari backend (GET /api/cctv/active)
2. Buka stream DroidCam/RTSP per kamera di background thread
3. Jalankan RT-DETR / YOLO inference setiap DETECTION_INTERVAL detik
4. Lacak objek dengan StationaryTracker (IoU)
5. Kirim hasil deteksi (status "abandoned" vs "tracking") ke backend (POST /api/detection)
"""

import time
import threading
import requests
from datetime import datetime, timezone
from ultralytics import YOLO, RTDETR

from config import (
    BACKEND_URL, AI_SERVER_SECRET,
    MODEL_PATH, CONFIDENCE_THRESHOLD,
    DETECTION_INTERVAL,
)
from utils import get_device
from camera import VideoStream
from layer1_tracker import StationaryTracker

# State per kamera: { cctv_id: { zona_id, detections, lock } }
camera_states: dict = {}

# ---------------------------------------------------------------------------
# Mapping kelas COCO → kategori sampah
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
# Per-camera Inference thread
# ---------------------------------------------------------------------------

def process_camera(cctv_id: str, stream_url: str, model, device: str):
    """Baca frame secara berkala (headless), inferensi, dan track dengan IoU."""
    print(f"[{cctv_id[:8]}] Mulai pemantauan (Headless) ke {stream_url} ...")
    cap = VideoStream(src=stream_url)
    time.sleep(1)  # tunggu buffer pertama

    last_predict_time = 0
    PREDICT_INTERVAL = DETECTION_INTERVAL
    
    # Init Tracker buat ngecek sampah yang udah diem 5 menit (300 detik)
    tracker = StationaryTracker(iou_threshold=0.45, max_missed=2, abandon_time_seconds=300)
    state = camera_states[cctv_id]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        current_time = time.time()

        # 1. Jalankan prediksi AI (Layer 2) hanya jika interval terjadwal terpenuhi
        if current_time - last_predict_time >= PREDICT_INTERVAL:
            results = model.predict(frame, device=device, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=320)
            
            raw_detections = []
            
            for box in results[0].boxes:
                cls_id     = int(box.cls[0])
                coco_label = model.names[cls_id]
                jenis      = TRASH_MAP.get(coco_label) or coco_label
                confidence = round(float(box.conf[0]), 4)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                raw_detections.append((x1, y1, x2, y2, jenis, confidence))

            # Teruskan ke Layer 1 (Math Tracker)
            tracked_objects = tracker.update(raw_detections)
            backend_valid_detections = []

            for obj in tracked_objects:
                # Kirim ke backend supaya dihitung / disimpan
                backend_valid_detections.append({
                    "jenis_objek": obj.jenis, 
                    "confidence": obj.confidence,
                    "status": "abandoned" if obj.is_abandoned else "tracking"
                })

            last_predict_time = current_time

            # Simpan deteksi untuk backend thread
            with state["lock"]:
                state["detections"] = backend_valid_detections

        # Waktu istirahat CPU minimum agar loop tidak 100% (karena cap.read instan dari background thread)
        time.sleep(0.05)


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
# Entry point
# ---------------------------------------------------------------------------

def main():
    global global_model, global_device
    print("=== CivicNode AI Server (Headless Data Node) ===")

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
        print("[server] Tidak ada kamera aktif saat startup. Akan tetap mengecek kamera secara dinamis.")

    # 3. Start detection sender thread
    t = threading.Thread(target=send_detections_loop, daemon=True)
    t.start()
    print(f"[server] Detections sender aktif, pooling tiap {DETECTION_INTERVAL} detik.")

    # 4. Daemon Loop
    print(f"[server] AI Server berjalan dalam mode Headless.")
    try:
        while True:
            time.sleep(60) # Cek kamera aktif baru tiap 60 detik
            sync_cameras()
            print(f"[server] Health check: {len(camera_states)} nodes active.")
    except KeyboardInterrupt:
        print("\n[server] Menghentikan server...")


if __name__ == "__main__":
    main()

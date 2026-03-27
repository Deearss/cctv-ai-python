# CivicNode AI Server (YOLOv11n)

Komponen AI untuk platform CivicNode — membaca stream kamera DroidCam, menjalankan deteksi sampah real-time dengan YOLO, lalu meneruskan hasilnya ke backend dan frontend.

---

## Arsitektur

```
DroidCam ──(MJPEG)──→ AI Server (server.py)
                            │
                            ├─(POST /api/detection tiap 1 detik)──→ Backend
                            │
                            └─(MJPEG stream ber-bounding-box)──→ Frontend
```

- AI server adalah **satu-satunya** yang connect ke DroidCam (DroidCam hanya support 1 koneksi MJPEG)
- Frontend tidak connect ke DroidCam langsung — stream diambil dari AI server
- Backend menerima data deteksi untuk disimpan ke DB dan ditampilkan di dashboard

---

## Tech Stack

- `Python 3.10+`
- `ultralytics` — YOLO inference
- `opencv-python` — baca & proses frame video
- `flask` — HTTP server + MJPEG streaming
- `requests` — komunikasi ke backend
- `PyTorch` — disarankan versi CUDA untuk performa optimal
- `python-dotenv` — konfigurasi lewat `.env`

---

## Setup

**1. Aktifkan virtual environment**
```bash
source bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
> Kalau pakai CUDA, install PyTorch manual sesuai instruksi di pytorch.org agar GPU terbaca.

**3. Konfigurasi `.env`**
```bash
cp .env.example .env
```
Isi nilainya:
```env
BACKEND_URL=http://localhost:3001
AI_SERVER_SECRET=<sama dengan yang di backend .env>
STREAM_PORT=5001
DETECTION_INTERVAL=1
```
> `URL_KAMERA` hanya diperlukan untuk `the_eye.py` dan `the_brain.py` (test mandiri).

---

## Cara Pakai

### Jalanin AI Server (integrasi penuh)
```bash
python server.py
```
Server akan:
1. Ambil daftar kamera aktif dari backend (`GET /api/cctv/active`)
2. Buka stream per kamera di background thread
3. Jalankan YOLO inference + gambar bounding box
4. Expose MJPEG stream di `http://0.0.0.0:5001/stream/<cctv_id>`
5. Kirim deteksi ke backend tiap `DETECTION_INTERVAL` detik

**Endpoints:**
- `GET /stream/<cctv_id>` — MJPEG stream ber-bounding-box
- `GET /health` — status server + daftar kamera aktif

---

### Test mandiri (tanpa backend)

**Test koneksi kamera + hardware:**
```bash
python the_eye.py
```

**Test YOLO inference lokal (tampil di window OpenCV):**
```bash
python the_brain.py
```

---

## Struktur File

| File | Fungsi |
|---|---|
| `server.py` | Entry point utama — Flask server + integrasi backend |
| `config.py` | Semua konfigurasi (URL, secret, threshold, port) |
| `camera.py` | `VideoStream` — baca frame kamera di background thread |
| `utils.py` | Deteksi hardware (CUDA / CPU fallback) |
| `the_eye.py` | Test koneksi kamera & hardware |
| `the_brain.py` | Test YOLO inference lokal (tanpa server) |
| `yolo11n.pt` | Model YOLO Nano (di-download otomatis kalau belum ada) |

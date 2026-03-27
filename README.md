# 👁️ CCTV AI Python (YOLOv11n)

Projek *Proof of Concept* (PoC) buat sistem CCTV cerdas berbasis AI yang jalan secara *real-time*. Di-support pakai **YOLOv11 Nano** buat *object detection* yang enteng tapi akurat, dan dibikin dengan arsitektur modular + *multithreading* buat narik video dari *IP Webcam/Smartphone* tanpa *bottleneck*. 

---

## 🚀 Fitur Utama

- **Real-time Object Detection**: Ngandelin model `YOLOv11n` dari Ultralytics, sanggup jalan mulus di Edge/GPU jadul kaya GTX 1050 Ti.
- **Multithreaded Camera Siphon**: *Feed* video ditarik di *background thread* pake `camera.py`. Main thread (AI) tinggal nyedot *frame* terbaru, *no lag, no stuttering*.
- **Config Driven**: Tinggal ganti IP Kamera lu atau *secret keys* lewat `.env` yang udah kepisah rapi sama source code. 
- **Auto Hardware Fallback**: Kalau Mesin/PC lu nggak punya CUDA/GPU NVIDIA, script bakal otomatis maksa *fallback* ke CPU (meski agak ngos-ngosan).

## 🛠️ Tech Stack & Prerequisite

- `Python 3.10+`
- `ultralytics` (YOLO)
- `opencv-python` (Visi Komputer & GUI)
- `PyTorch` (Disarankan yg *CUDA enabled*)
- `python-dotenv`

## ⚙️ Cara Setup

1. **Clone Repo**
   ```bash
   git clone https://github.com/Deearss/cctv-ai-python.git
   cd venv_cctv
   ```

2. **Setup Environtment Variable**
   Kopi file *example* ke ke konfigurasi rill lu:
   ```bash
   cp .env.example .env
   ```
   *(Note: Buka `.env` terus set atau timpa nilai `URL_KAMERA` sesuai IP dari aplikasi IP Webcam di HP lu!)*

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Penting: kalau lu pakai CUDA, mending pastiin Pytorch diinstall manual ngikutin instruksi situs resminya biar GPU lu kebaca).*

## 🎮 Cara Pakai

Gue mecah filenya biar lebih enak di-test satu-satu:

### 1. Test Hardware & Kamera (The Eye)
Jalanin skrip ini buat ngecek GPU dan mastiin HP lu beneran nyambung lewat WiFi (nggak dipakein AI dulu).
```bash
python the_eye.py
```

### 2. Mulai AI CCTV (The Brain)
Skrip utama buat deteksi AI! Ini bakal download `yolo11n.pt` otomatis (kalau belum ada) terus nampilin *inferencing* secara *real-time*.
```bash
python the_brain.py
```

## 📂 Struktur Repositori

- **`config.py`**: Jantung konfigurasi, ngatur IP Webcam, *keys*, dan *threshold* AI.
- **`utils.py`**: Fungsi rapih buat nyari dan nentuin kapabilitas perangkat keras.
- **`camera.py`**: Kelas magis (`VideoStream`) buat baca feed video asinkronus (*multithreading*).
- **`the_eye.py`**: Skrip diagnostik *streaming* dan *hardware test*.
- **`the_brain.py`**: Modul pusat, integrasi model AI ke *loop* visual.

<br/>

> *Built with systems design mindset for maintainability and scalability.* 🗿

import os
from dotenv import load_dotenv

# Load environment variable dari file .env
load_dotenv()

# IP Webcam lu -> Ganti di sini kalau IP nya berubah.
# Pake Environment Variable kalau pengen lewat terminal, default ke IP ini.
URL_KAMERA = os.getenv("URL_KAMERA", "http://192.168.18.3:4747/video")

# Path ke file model lu
MODEL_PATH = "yolo11n.pt"

# Batas minimum probabilitas objek dideteksi
CONFIDENCE_THRESHOLD = 0.5

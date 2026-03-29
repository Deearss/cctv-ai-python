import os
from dotenv import load_dotenv

load_dotenv()

# Backend
BACKEND_URL       = os.getenv("BACKEND_URL", "http://localhost:3001")
AI_SERVER_SECRET  = os.getenv("AI_SERVER_SECRET", "")

# URL kamera — masih dipakai the_eye.py dan the_brain.py untuk test mandiri
URL_KAMERA = os.getenv("URL_KAMERA", "http://192.168.18.3:4747/video")

# YOLO / RT-DETR
MODEL_PATH           = "rtdetr-l.pt"
CONFIDENCE_THRESHOLD = 0.5

# Server
STREAM_PORT        = int(os.getenv("STREAM_PORT", "5001"))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "60"))  # detik

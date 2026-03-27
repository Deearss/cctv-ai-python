import cv2
import time
from ultralytics import YOLO
from config import URL_KAMERA, MODEL_PATH, CONFIDENCE_THRESHOLD
from utils import get_device
from camera import VideoStream

def main():
    print("Loading AI Model...")
    model = YOLO(MODEL_PATH) 
    
    device = get_device()
    print(f"AI bakal jalan di: {device}\n")

    print("Mulai stream kamera...")
    cap = VideoStream(src=URL_KAMERA)
    time.sleep(1) # Tunggu buffer pertama

    while True:
        ret, frame = cap.read()
        
        # Lewatin frame jelek/kosong
        if not ret or frame is None:
            continue

        # PROSES AI (Inference)
        results = model.predict(frame, device=device, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Gambar deteksi ke frame
        annotated_frame = results[0].plot()

        cv2.imshow("CCTV AI - The Brain", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
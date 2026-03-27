import cv2
import sys
import time
from config import URL_KAMERA
from utils import get_device
from camera import VideoStream

def main():
    device = get_device()
    print(f"\nNyoba konek ke {URL_KAMERA}...")
    
    # Init kamera pake background thread
    cap = VideoStream(src=URL_KAMERA)
    time.sleep(1) # Kasih waktu dikit buat buffer pertama masuk
    
    ret, frame = cap.read()
    if not cap.stream.isOpened() or not ret:
        print(f"Error: Gagal tarik gambar dari {URL_KAMERA}. Cek WiFi/IP Webcam lu!")
        cap.stop()
        sys.exit()

    print(f"Koneksi sukses (Jalan di {device})! Tekan 'q' buat keluar.")

    while True:
        ret, frame = cap.read()
        
        # Kalau koneksi skip frame / lemot, lanjut aja jangan crash
        if not ret or frame is None:
            continue

        cv2.imshow("CCTV AI - The Eye (Test Mode)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
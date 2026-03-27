import cv2
import threading

class VideoStream:
    """Class ini narik frame video di background thread biar AI/Inference
    di main thread nggak nungguin proses I/O network yang lemot.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        
        # Mulai thread buat nyedot frame terus-terusan di background
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Terus-terusan jalan sampe method stop() dipanggil
        while True:
            if self.stopped:
                return
            
            # Selalu tiban dengan frame terbaru, gak usah ngantri buffer panjang
            self.ret, self.frame = self.stream.read()

    def read(self):
        # Balikin state terbaru dari stream
        return self.ret, self.frame

    def stop(self):
        # Stop thread perlahan dan lepas koneksi kamera
        self.stopped = True
        self.thread.join()
        self.stream.release()

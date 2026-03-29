import time

def compute_iou(boxA, boxB):
    """Hitung Intersection over Union (IoU) dari dua bounding box (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class TrackedObject:
    def __init__(self, obj_id, box, jenis, confidence):
        self.id = obj_id
        self.box = box
        self.jenis = jenis
        self.confidence = confidence
        
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.hit_count = 1       # Berapa kali terdeteksi berturut-turut (atau total)
        self.missed_count = 0    # Berapa cycle dia hilang
        self.is_abandoned = False

    def update(self, box, confidence):
        self.box = box           # Update lokasi ke yang terbaru
        self.confidence = confidence
        self.last_seen = time.time()
        self.hit_count += 1
        self.missed_count = 0    # Reset miss count karena barusan keliatan lagi

class StationaryTracker:
    def __init__(self, iou_threshold=0.45, max_missed=2, abandon_time_seconds=300):
        """
        iou_threshold: Seberapa ketat overlapping box yang dianggap objek yang sama
        max_missed: Boleh ilang/gak ke-deteksi di n cycle sebelum dihapus dari tracker
        abandon_time_seconds: Waktu (detik) sampai objek dinobatkan sebagai 'Valid Abandoned'
        """
        self.tracked_objects = []
        self.next_obj_id = 1
        
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.abandon_time_seconds = abandon_time_seconds

    def update(self, new_detections):
        """
        new_detections list of tuple: [(x1, y1, x2, y2, jenis, confidence), ...]
        """
        # 1. Tandai semua objek "missed" sementara
        for obj in self.tracked_objects:
            obj.missed_count += 1

        # 2. Match deteksi baru dengan objek yang lagi di-track
        for new_det in new_detections:
            new_box = new_det[:4]
            jenis = new_det[4]
            conf = new_det[5]

            best_match = None
            best_iou = self.iou_threshold

            for obj in self.tracked_objects:
                if obj.jenis == jenis:  # Harus objek sejenis
                    iou = compute_iou(obj.box, new_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj

            if best_match is not None:
                # Update objek lama
                best_match.update(new_box, conf)
                # Kurangi missed_count yang tadi ditambahin di awal
                best_match.missed_count -= 1
            else:
                # Daftarin objek 100% baru
                new_obj = TrackedObject(self.next_obj_id, new_box, jenis, conf)
                self.tracked_objects.append(new_obj)
                self.next_obj_id += 1

        # 3. Hapus objek yang kelamaan hilang
        self.tracked_objects = [obj for obj in self.tracked_objects if obj.missed_count <= self.max_missed]

        # 4. Cek mana yang udah jadi 'Abandoned' (diam > abandon_time_seconds)
        current_time = time.time()
        for obj in self.tracked_objects:
            if not obj.is_abandoned:
                if (current_time - obj.first_seen) >= self.abandon_time_seconds:
                    obj.is_abandoned = True

        return self.tracked_objects

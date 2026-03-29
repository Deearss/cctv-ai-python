# Progress Log

## Sesi 2026-03-29

### Yang Sudah Selesai
- ✅ **Layer 1 & Layer 2 Tracking Architecture** — Memecah deteksi AI menjadi dua lapis. Layer 2 (`rtdetr-l.pt`) menembak *inference* menggunakan beban GPU per 60 detik. Layer 1 (`layer1_tracker.py` dengan objek `StationaryTracker`) menggunakan kalkulasi IoU (Intersection-over-Union) secara berkelanjutan dengan batas *abandon_time* 300 detik.
- ✅ **Headless Data Node Transformation** — Membuang Flask, _mjpeg feed_, manipulasi frame `.jpg`, dan anotasi `cv2.rectangle` sepenuhnya dari `server.py`. Skrip diringankan menjadi pure headless-daemon-worker untuk mencegah pemborosan VRAM & CPU dalam menyajikan *user interface* (karena _frontend stream_ layar dihapus). Payload ke Backend diubah orientasinya ke *Data Analytics* bukan Video Visual.
- ✅ **Deteksi "Semua Objek"** — Menonaktifkan sensor filter `TRASH_MAP` (tetap dipakai, namun `fallback` digunakan untuk mendeteksi `car`, `person`, `motorcycle` dsb, untuk simulasi hit rate deteksi.

### Yang Menunggu Pengerjaan / Blocked
- ⏳ **Restrukturisasi Payload API Detections** — Saat ini POST payload yang dikirim dari `venv_cctv/server.py` masih memakai struktur JSON lama berformat array koordinat detektor. Menunggu hasil *brainstorming Database Schema* bersama agen AI (NotebookLM) / owner untuk merubah *value streaming* tersebut menjadi agregat yang kompatibel dengan Analytics Timeline Frontend.

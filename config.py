"""
config.py — Parameter dan threshold sistem loitering detection.

DEFINISI LOITERING (Konteks Bank ATM):

  LOITERING (definisi utama = mondar-mandir / pacing):
    - Mondar-mandir (pacing) >= PACING_DIRECTION_CHANGE_THRESHOLD kali
    Heuristik pendukung (opsional, masih aktif di kode):
    - Masuk-keluar berulang >= REENTRY_COUNT_THRESHOLD dalam REENTRY_WINDOW_SEC
    - >= CROWD_COUNT_THRESHOLD orang berkerumun di depan satu ATM

  TIDAK dianggap loitering:
    - Orang yang datang, langsung transaksi, lalu pergi (< 45 detik)
    - Orang mengantri di belakang zona ATM
    - Satpam yang berpatroli normal (akan pacing tapi keluar-masuk ROI cepat)
"""

# ROI (Region of Interest)
# Format: (x1, y1, x2, y2) piksel, atau None = seluruh frame
ATM_ROI = None

# THRESHOLD WAKTU
# 45 detik: transaksi ATM normal ~20-40 detik, beri toleransi
LOITERING_TIME_THRESHOLD_SEC = 60.0

# 25 detik diam: orang yang input PIN + menunggu wajar diam ~15 detik
# Lebih dari 25 detik diam = mencurigakan
IDLE_TIME_THRESHOLD_SEC = 120.0
 
# THRESHOLD PERGERAKAN
MOVEMENT_MIN_DISTANCE_PX = 15
SPEED_WINDOW_FRAMES = 10
IDLE_SPEED_THRESHOLD = 2.0

 
# PACING (mondar-mandir) 
# 6 pergantian arah: memastikan benar-benar mondar-mandir, bukan sekadar
# bergeser sedikit saat transaksi
PACING_DIRECTION_CHANGE_THRESHOLD = 6
PACING_WINDOW_FRAMES = 60

# CROWD DETECTION
 
CROWD_COUNT_THRESHOLD = 2

# RE-ENTRY DETECTION
 
REENTRY_COUNT_THRESHOLD = 2
REENTRY_WINDOW_SEC = 120.0
EXIT_CONFIRMATION_SEC = 2.0


# FACE COVERING
 
HEAD_AREA_RATIO_THRESHOLD = 1.5  # rasio area bbox kepala vs tubuh (helm/masker/hoodie bisa lebih besar)

 
# TRACKING
 
# 60 frame: ByteTrack lebih sabar sebelum drop ID (kurangi flickering)
TRACK_LOST_FRAMES = 60
YOLO_CONFIDENCE = 0.45
TRACK_IOU_THRESHOLD = 0.3

# MODEL
YOLO_MODEL = "yolov8n.pt"

# OUTPUT
OUTPUT_DIR = "output"
SAVE_SCREENSHOTS = True
SAVE_VIDEO = True
SHOW_VIDEO = True

# VISUALISASI
COLOR_NORMAL    = (0, 200, 0)       # hijau
COLOR_WARNING   = (0, 165, 255)     # oranye
COLOR_LOITERING = (0, 0, 220)       # merah (alert loitering)
COLOR_ROI       = (255, 255, 0)     # kuning (garis zona ATM)

WARNING_THRESHOLD_RATIO = 0.7

# LOGGING
LOG_FILE = "loitering_log.csv"

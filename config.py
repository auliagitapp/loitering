"""
config.py — Parameter dan threshold sistem loitering detection.

DEFINISI LOITERING (Konteks Bank ATM):

  LEVEL 1 — LOITERING (oranye/merah):
    Perilaku mencurigakan yang butuh observasi.
    - Berada di zona ATM > LOITERING_TIME_THRESHOLD_SEC tanpa transaksi nyata
    - Berdiri diam > IDLE_TIME_THRESHOLD_SEC (kemungkinan pasang skimmer diam-diam)
    - Mondar-mandir (pacing) >= PACING_DIRECTION_CHANGE_THRESHOLD kali
    - Masuk-keluar berulang >= REENTRY_COUNT_THRESHOLD dalam REENTRY_WINDOW_SEC
    - >= CROWD_COUNT_THRESHOLD orang berkerumun di depan satu ATM
    - Kemungkinan wajah tertutup (helm/masker/hoodie)

  LEVEL 2 — TAMPERING / CRITICAL (magenta):
    Tindakan yang hampir pasti merupakan kejahatan.
    - Posisi tubuh ekstrem: membungkuk sangat dalam ke mesin (bbox melebar)
      → bbox width/height ratio > TAMPERING_POSE_RATIO_THRESHOLD
    - Lonjakan gerakan mendadak (memukul/merusak mesin)
      → optical flow z-score > TAMPERING_MOTION_ZSCORE_THRESHOLD

  TIDAK dianggap loitering:
    - Orang yang datang, langsung transaksi, lalu pergi (< 45 detik)
    - Orang mengantri di belakang zona ATM
    - Satpam yang berpatroli normal (akan pacing tapi keluar-masuk ROI cepat)
"""

# =============================================================================
# ROI (Region of Interest)
# Format: (x1, y1, x2, y2) piksel, atau None = seluruh frame
# =============================================================================
ATM_ROI = None

# =============================================================================
# THRESHOLD WAKTU
# =============================================================================
# 45 detik: transaksi ATM normal ~20-40 detik, beri toleransi
LOITERING_TIME_THRESHOLD_SEC = 45.0

# 25 detik diam: orang yang input PIN + menunggu wajar diam ~15 detik
# Lebih dari 25 detik diam = mencurigakan
IDLE_TIME_THRESHOLD_SEC = 25.0

# =============================================================================
# THRESHOLD PERGERAKAN
# =============================================================================
MOVEMENT_MIN_DISTANCE_PX = 15
SPEED_WINDOW_FRAMES = 10
IDLE_SPEED_THRESHOLD = 2.0

# =============================================================================
# PACING (mondar-mandir)
# =============================================================================
# 6 pergantian arah: memastikan benar-benar mondar-mandir, bukan sekadar
# bergeser sedikit saat transaksi
PACING_DIRECTION_CHANGE_THRESHOLD = 6
PACING_WINDOW_FRAMES = 60

# =============================================================================
# CROWD DETECTION
# =============================================================================
CROWD_COUNT_THRESHOLD = 2

# =============================================================================
# RE-ENTRY DETECTION
# =============================================================================
REENTRY_COUNT_THRESHOLD = 2
REENTRY_WINDOW_SEC = 120.0
EXIT_CONFIRMATION_SEC = 2.0

# =============================================================================
# TAMPERING DETECTION (Level 2 — KRITIS)
# =============================================================================
# Aspect ratio bbox: orang berdiri normal ~0.3-0.5, membungkuk dalam ~0.7+
# Contoh: orang pegang linggis + membungkuk ke ATM bbox-nya hampir persegi
TAMPERING_POSE_RATIO_THRESHOLD = 0.85

# Z-score optical flow: berapa SD di atas rata-rata agar dianggap lonjakan
# Nilai 13 = sangat yakin ada lonjakan mendadak (memukul/merusak)
TAMPERING_MOTION_ZSCORE_THRESHOLD = 13

# Minimum magnitude absolut untuk tampering motion (filter noise)
TAMPERING_MOTION_MIN_MAGNITUDE = 6.0

# =============================================================================
# FACE COVERING
# =============================================================================
HEAD_AREA_RATIO_THRESHOLD = 0.15

# =============================================================================
# TRACKING
# =============================================================================
# 60 frame: ByteTrack lebih sabar sebelum drop ID (kurangi flickering)
TRACK_LOST_FRAMES = 60
YOLO_CONFIDENCE = 0.45
TRACK_IOU_THRESHOLD = 0.3

# =============================================================================
# MODEL
# =============================================================================
YOLO_MODEL = "yolov8n.pt"

# =============================================================================
# OUTPUT
# =============================================================================
OUTPUT_DIR = "output"
SAVE_SCREENSHOTS = True
SAVE_VIDEO = True
SHOW_VIDEO = True

# =============================================================================
# VISUALISASI
# =============================================================================
COLOR_NORMAL    = (0, 200, 0)       # hijau
COLOR_WARNING   = (0, 165, 255)     # oranye
COLOR_LOITERING = (0, 0, 220)       # merah (Level 1)
COLOR_TAMPERING = (180, 0, 255)     # magenta terang (Level 2 — KRITIS)
COLOR_ROI       = (255, 255, 0)     # kuning (garis zona ATM)

WARNING_THRESHOLD_RATIO = 0.7

# =============================================================================
# LOGGING
# =============================================================================
LOG_FILE = "loitering_log.csv"

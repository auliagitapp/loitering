"""
tracker.py — State management per individu dengan dua level alert.

DEFINISI ULANG untuk konteks Bank ATM:

LEVEL 1 — LOITERING (kuning/oranye):
  Perilaku mencurigakan yang butuh observasi lebih lanjut.
  - Berada di zona ATM > 45 detik tanpa bergerak ke arah mesin
  - Mondar-mandir (pacing) berulang kali
  - Masuk-keluar zona berulang dalam waktu singkat
  - Lebih dari 3 orang berkerumun di depan satu ATM

LEVEL 2 — TAMPERING / CRITICAL (merah terang):
  Tindakan yang hampir pasti merupakan kejahatan/vandalisme.
  - Posisi tubuh ekstrem: membungkuk sangat dalam ke mesin (aspect ratio bbox melebar)
  - Gerakan ayun/pukulan: lonjakan motion magnitude tiba-tiba yang tinggi
  - Diam total di depan ATM terlalu lama (bisa sedang memasang skimmer)

FIX detection hilang saat diam:
  Detection YOLOv8 kadang drop saat orang sangat diam (confidence turun).
  Solusi: grace period diperpanjang + interpolasi posisi terakhir.
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import config


@dataclass
class LoiteringFlags:
    # Level 1 — Loitering
    time_exceeded: bool = False
    idle_too_long: bool = False
    pacing: bool = False
    crowding: bool = False
    reentry: bool = False
    face_covered: bool = False
    # Level 2 — Tampering/Critical
    tampering_pose: bool = False   # posisi tubuh ekstrem (membungkuk ke mesin)
    tampering_motion: bool = False # lonjakan gerakan mendadak (memukul/merusak)

    def any(self) -> bool:
        return any([
            self.time_exceeded, self.idle_too_long, self.pacing,
            self.crowding, self.reentry, self.face_covered,
            self.tampering_pose, self.tampering_motion,
        ])

    def is_critical(self) -> bool:
        """Level 2: tampering yang butuh respons segera."""
        return self.tampering_pose or self.tampering_motion

    def describe(self) -> str:
        reasons = []
        # Critical dulu agar muncul di depan
        if self.tampering_pose:   reasons.append("TAMPER-POSE")
        if self.tampering_motion: reasons.append("TAMPER-HIT")
        if self.time_exceeded:    reasons.append("TIME")
        if self.idle_too_long:    reasons.append("IDLE")
        if self.pacing:           reasons.append("PACING")
        if self.crowding:         reasons.append("CROWD")
        if self.reentry:          reasons.append("REENTRY")
        if self.face_covered:     reasons.append("FACE?")
        return "+".join(reasons) if reasons else ""


class PersonState:
    """
    State satu individu. Mendukung:
    - Inherited time dari ID sebelumnya (fix flickering)
    - Dua level alert (loitering vs tampering)
    - Grace period panjang agar deteksi tidak hilang saat orang diam
    """

    MIN_FRAMES_BEFORE_FLAG = 15  # Tunggu stabilisasi sebelum flag apapun

    def __init__(self, track_id: int, fps: float, inherited_time: float = 0.0):
        self.track_id = track_id
        self.fps = fps

        self.time_in_roi: float = inherited_time
        self.first_seen: float = time.time()
        self.last_seen: float = time.time()

        self.positions: deque = deque(maxlen=300)
        self.bbox_history: deque = deque(maxlen=60)
        self.last_bbox: Optional[tuple] = None

        # Motion history untuk deteksi lonjakan mendadak (tampering hit)
        self.motion_magnitudes: deque = deque(maxlen=30)
        self.speeds: deque = deque(maxlen=config.SPEED_WINDOW_FRAMES)
        self.last_significant_move_time: float = time.time()

        # Pacing
        self.direction_history: deque = deque(maxlen=config.PACING_WINDOW_FRAMES)
        self.last_x: Optional[float] = None
        self.direction_changes: int = 0

        # Re-entry
        self.entry_times: list = []
        self.exit_times: list = []
        self.currently_in_roi: bool = True

        self.flags = LoiteringFlags()
        self.is_loitering: bool = False
        self.alert_sent: bool = False
        self._frame_count: int = 0

    def update(self, bbox: tuple, in_roi: bool, current_time: float,
               motion_magnitude: float = 0.0):
        """
        Update state dengan deteksi baru.

        Args:
            bbox             : (x1,y1,x2,y2)
            in_roi           : apakah dalam zona ATM
            current_time     : detik dari awal video
            motion_magnitude : rata-rata optical flow magnitude di area bbox (dari OpticalFlow)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self.last_bbox = bbox
        self._frame_count += 1

        self.bbox_history.append(bbox)
        self.motion_magnitudes.append(motion_magnitude)

        if in_roi:
            if self.last_seen > 0:
                delta = current_time - self.last_seen
                if 0 < delta < 5.0:
                    self.time_in_roi += delta

            self.last_seen = current_time
            self.positions.append((cx, cy))

            if len(self.positions) >= 2:
                prev = self.positions[-2]
                dist = np.sqrt((cx - prev[0])**2 + (cy - prev[1])**2)
                self.speeds.append(dist)
                if dist > config.MOVEMENT_MIN_DISTANCE_PX:
                    self.last_significant_move_time = current_time

            self._update_pacing(cx)

            if not self.currently_in_roi:
                self.currently_in_roi = True
                self.entry_times.append(current_time)
        else:
            if self.currently_in_roi:
                self.currently_in_roi = False
                self.exit_times.append(current_time)

        self._evaluate_flags(current_time)

    def _update_pacing(self, cx: float):
        if self.last_x is None:
            self.last_x = cx
            return
        delta_x = cx - self.last_x
        if abs(delta_x) < config.MOVEMENT_MIN_DISTANCE_PX:
            return
        current_dir = 1 if delta_x > 0 else -1
        if self.direction_history and self.direction_history[-1] != current_dir:
            self.direction_changes += 1
        self.direction_history.append(current_dir)
        self.last_x = cx
        if len(self.direction_history) == config.PACING_WINDOW_FRAMES:
            changes = sum(
                1 for i in range(1, len(self.direction_history))
                if self.direction_history[i] != self.direction_history[i-1]
            )
            self.direction_changes = changes

    def _evaluate_flags(self, current_time: float):
        if self._frame_count < self.MIN_FRAMES_BEFORE_FLAG:
            return

        # DINONAKTIFKAN — menyebabkan false positive untuk transaksi normal:
        # Orang diam di ATM, antri, atau satpam patroli bukan berarti merambok.
        self.flags.time_exceeded = False
        self.flags.idle_too_long = False
        self.flags.pacing        = False

        # AKTIF — indikator perampokan/tampering:

        # Re-entry: keluar-masuk berulang = survei/intai lokasi sebelum beraksi
        recent_entries = [
            t for t in self.entry_times
            if current_time - t <= config.REENTRY_WINDOW_SEC
        ]
        self.flags.reentry = len(recent_entries) >= config.REENTRY_COUNT_THRESHOLD

        # Face covered: helm/masker/hoodie = sengaja sembunyikan identitas
        self.flags.face_covered = self._check_face_covered()

        # Crowding: komplotan (diset dari TrackManager.update_tracks)

        # Tampering pose: bbox melebar = membungkuk dalam / mengayun objek ke mesin
        self.flags.tampering_pose = self._check_tampering_pose()

        # Tampering motion: lonjakan optical flow mendadak = memukul/merusak ATM
        self.flags.tampering_motion = self._check_tampering_motion()

        self.is_loitering = self.flags.any()

    def _check_face_covered(self) -> bool:
        if len(self.bbox_history) < 5:
            return False
        recent = list(self.bbox_history)[-5:]
        heights = [(b[3] - b[1]) for b in recent]
        widths  = [(b[2] - b[0]) for b in recent]
        avg_h = np.mean(heights)
        avg_w = np.mean(widths)
        if avg_h < 50:
            return False
        return (avg_w / avg_h) > 0.55

    def _check_tampering_pose(self) -> bool:
        """
        Deteksi posisi tubuh ekstrem: membungkuk dalam ke mesin ATM.

        Saat orang membungkuk sangat dalam, bounding box mereka menjadi
        jauh lebih lebar dibanding tingginya (aspect ratio > threshold).
        Ini berbeda dari orang berdiri normal (aspect ratio ~0.3-0.5).

        Contoh dari screenshot: orang pegang linggis + membungkuk →
        bbox hampir persegi atau lebih lebar dari tinggi.
        """
        if len(self.bbox_history) < 10:
            return False

        # Gunakan median 10 frame terakhir agar tidak sensitif terhadap 1 frame aneh
        recent = list(self.bbox_history)[-10:]
        ratios = [(b[2]-b[0]) / max(b[3]-b[1], 1) for b in recent]
        median_ratio = float(np.median(ratios))

        # Normal berdiri: ~0.3-0.5
        # Membungkuk dalam / mengayun objek: > 0.7
        return median_ratio > config.TAMPERING_POSE_RATIO_THRESHOLD

    def _check_tampering_motion(self) -> bool:
        """
        Deteksi lonjakan gerakan mendadak yang tinggi.

        Memukul ATM dengan objek → optical flow magnitude di area bbox
        akan melonjak jauh di atas rata-rata normal.
        Gunakan z-score: apakah magnitude saat ini jauh di atas rata-rata historis?
        """
        if len(self.motion_magnitudes) < 10:
            return False

        mags = np.array(list(self.motion_magnitudes))
        mean = mags.mean()
        std  = mags.std()

        if std < 0.1:  # std terlalu kecil, tidak ada variasi
            return False

        # Z-score frame terakhir vs historis
        latest = mags[-1]
        z_score = (latest - mean) / std

        return (z_score > config.TAMPERING_MOTION_ZSCORE_THRESHOLD
                and latest > config.TAMPERING_MOTION_MIN_MAGNITUDE)

    @property
    def avg_speed(self) -> float:
        if not self.speeds:
            return 0.0
        return float(np.mean(self.speeds))

    def get_status_color(self) -> tuple:
        """
        Tiga level warna:
          Hijau  = normal
          Oranye = warning (mendekati threshold loitering)
          Merah  = loitering Level 1
          Merah terang/magenta = tampering Level 2 (KRITIS)
        """
        if self.flags.is_critical():
            return config.COLOR_TAMPERING  # merah terang / magenta
        if self.is_loitering:
            return config.COLOR_LOITERING
        ratio = self.time_in_roi / config.LOITERING_TIME_THRESHOLD_SEC
        if ratio >= config.WARNING_THRESHOLD_RATIO:
            return config.COLOR_WARNING
        return config.COLOR_NORMAL


def _bbox_iou(b1: tuple, b2: tuple) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


class TrackManager:
    """
    Mengelola semua PersonState aktif.

    FIX detection hilang saat diam:
    Saat YOLOv8 drop deteksi (confidence turun karena orang diam),
    kita tidak langsung hapus track — gunakan GHOST_FRAMES_THRESHOLD
    yang lebih panjang, dan pertahankan bbox terakhir di posisi yang sama.
    """

    SAME_PERSON_IOU_THRESHOLD = 0.3
    # Ghost threshold dalam DETIK (dikonversi ke frame saat runtime)
    # Konsisten untuk semua FPS — 3 detik cukup untuk recovery occlusion singkat
    GHOST_SEC_THRESHOLD = 3.0

    def __init__(self, fps: float):
        self.fps = fps
        self.active_tracks: dict[int, PersonState] = {}
        self.archived_tracks: dict[int, PersonState] = {}
        self._alerted_cells: set = set()
        self._alert_cooldown: dict[int, float] = {}  # track_id → last alert timestamp
        # Frame counter untuk ghost tracking
        self._ghost_counters: dict[int, int] = {}

    def update_tracks(self, detections: list, in_roi_flags: list,
                      current_time: float, motion_map: np.ndarray = None):
        """
        Args:
            detections  : list of (track_id, bbox)
            in_roi_flags: list of bool
            current_time: detik dari awal video
            motion_map  : optional — grayscale magnitude map dari optical flow,
                          digunakan untuk menghitung motion per bbox
        """
        active_ids = set()

        for (track_id, bbox), in_roi in zip(detections, in_roi_flags):
            active_ids.add(track_id)
            self._ghost_counters[track_id] = 0  # reset ghost counter

            if track_id not in self.active_tracks:
                inherited_time = self._find_inherited_time(bbox)
                self.active_tracks[track_id] = PersonState(
                    track_id, self.fps, inherited_time=inherited_time
                )

            # Hitung motion magnitude di area bbox jika ada motion_map
            mag = self._get_bbox_motion(bbox, motion_map) if motion_map is not None else 0.0
            self.active_tracks[track_id].update(bbox, in_roi, current_time, mag)

        # Handle track yang tidak muncul di frame ini
        missing_ids = set(self.active_tracks.keys()) - active_ids
        lost_threshold_sec = config.TRACK_LOST_FRAMES / self.fps

        for tid in list(missing_ids):
            person = self.active_tracks[tid]
            time_since_last = current_time - person.last_seen

            # Ghost tracking: pertahankan di posisi terakhir sebelum benar-benar hapus
            # Ini mencegah deteksi "hilang" saat orang diam sebentar
            self._ghost_counters[tid] = self._ghost_counters.get(tid, 0) + 1

            ghost_frame_limit = int(self.GHOST_SEC_THRESHOLD * self.fps)
            if self._ghost_counters[tid] < ghost_frame_limit:
                # Masih dalam grace period — update dengan bbox terakhir yang diketahui
                if person.last_bbox is not None:
                    mag = self._get_bbox_motion(person.last_bbox, motion_map) \
                          if motion_map is not None else 0.0
                    person.update(person.last_bbox, person.currently_in_roi,
                                  current_time, mag)
            elif time_since_last > lost_threshold_sec:
                # Benar-benar hilang — arsipkan
                self.archived_tracks[tid] = self.active_tracks.pop(tid)
                self._ghost_counters.pop(tid, None)

        # Bersihkan archived yang sudah > 10 detik
        for tid in list(self.archived_tracks.keys()):
            if current_time - self.archived_tracks[tid].last_seen > 10.0:
                del self.archived_tracks[tid]

        # Crowd detection
        in_roi_count = sum(
            1 for p in self.active_tracks.values() if p.currently_in_roi
        )
        for person in self.active_tracks.values():
            person.flags.crowding = (
                person.currently_in_roi
                and in_roi_count >= config.CROWD_COUNT_THRESHOLD
            )
            person.is_loitering = person.flags.any()

    def _get_bbox_motion(self, bbox: tuple, motion_map: np.ndarray) -> float:
        """Hitung rata-rata motion magnitude di dalam area bounding box."""
        if motion_map is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        h, w = motion_map.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w-1, x2); y2 = min(h-1, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = motion_map[y1:y2, x1:x2]
        return float(region.mean()) if region.size > 0 else 0.0

    def _find_inherited_time(self, new_bbox: tuple) -> float:
        best_iou = self.SAME_PERSON_IOU_THRESHOLD
        best_time = 0.0
        for archived in self.archived_tracks.values():
            if archived.last_bbox is None:
                continue
            iou = _bbox_iou(new_bbox, archived.last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_time = archived.time_in_roi
        return best_time

    def should_send_alert(self, person: PersonState) -> bool:
        """
        Cegah alert spam.
        Grid 200px + cooldown 10 detik per track_id.
        Untuk video FPS rendah / slow motion, orang yang sama
        bergerak lambat tapi ID ganti-ganti → pakai grid lebih besar.
        """
        if not person.positions:
            return False

        import time as _time

        # Cooldown per track_id: jangan alert ID yang sama dalam 10 detik
        now = _time.time()
        last_alert = self._alert_cooldown.get(person.track_id, 0)
        if now - last_alert < 10.0:
            return False

        cx, cy = person.positions[-1]
        # Grid 200px — lebih toleran terhadap pergerakan kecil
        cell = (int(cx // 200), int(cy // 200), person.flags.describe())
        if cell in self._alerted_cells:
            return False

        self._alerted_cells.add(cell)
        self._alert_cooldown[person.track_id] = now
        return True

    def get_loitering_persons(self) -> list:
        return [p for p in self.active_tracks.values() if p.is_loitering]

    def get_all_active(self) -> list:
        return list(self.active_tracks.values())

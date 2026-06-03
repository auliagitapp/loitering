"""
utils.py — Fungsi-fungsi pendukung: load anotasi, pilih ROI, logging, screenshot.
"""

import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# ANNOTATION LOADER
# =============================================================================

def load_annotations(csv_path: str) -> dict:
    """
    Load anotasi Kaggle dari file CSV.
    
    Format yang didukung (dari contoh yang kamu berikan):
        filename | total_frames | fps | loiter_start | loiter_end
        3.mp4    | 737          | 30  | 1            | 737
        12.mp4   | 910          | 24  | -1           | -1       ← no loitering

    Returns:
        dict {filename: {total_frames, fps, loiter_start, loiter_end}}
    """
    annotations = {}

    if not os.path.exists(csv_path):
        print(f"[WARN] File anotasi tidak ditemukan: {csv_path}")
        return annotations

    with open(csv_path, "r") as f:
        # Coba baca header dulu
        sample = f.read(512)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)

        reader = csv.reader(f, delimiter="\t") if "\t" in sample else csv.reader(f)

        for i, row in enumerate(reader):
            if has_header and i == 0:
                continue
            if len(row) < 5:
                # Coba split by whitespace jika bukan CSV biasa
                row = row[0].split() if row else []
            if len(row) < 5:
                continue

            try:
                fname        = row[0].strip()
                total_frames = int(row[1])
                fps          = float(row[2])
                loiter_start = int(row[3])
                loiter_end   = int(row[4])

                annotations[fname] = {
                    "total_frames": total_frames,
                    "fps": fps,
                    "loiter_start": loiter_start,
                    "loiter_end": loiter_end,
                    "has_loitering": loiter_start != -1,
                }
            except (ValueError, IndexError) as e:
                print(f"[WARN] Baris anotasi tidak valid (row {i}): {row} — {e}")

    print(f"[INFO] Loaded {len(annotations)} anotasi")
    for fname, ann in annotations.items():
        status = f"loitering frame {ann['loiter_start']}–{ann['loiter_end']}" \
                 if ann['has_loitering'] else "no loitering"
        print(f"  {fname:15s}  {ann['total_frames']:5d} frames  {ann['fps']} fps  → {status}")

    return annotations


# =============================================================================
# ROI SELECTOR
# =============================================================================

_roi_drawing = False
_roi_start = (0, 0)
_roi_end = (0, 0)
_roi_confirmed = False
_roi_result = None


def _roi_mouse_callback(event, x, y, flags, param):
    global _roi_drawing, _roi_start, _roi_end, _roi_confirmed, _roi_result

    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_drawing = True
        _roi_start = (x, y)
        _roi_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and _roi_drawing:
        _roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        _roi_drawing = False
        _roi_end = (x, y)
        x1, y1 = min(_roi_start[0], _roi_end[0]), min(_roi_start[1], _roi_end[1])
        x2, y2 = max(_roi_start[0], _roi_end[0]), max(_roi_start[1], _roi_end[1])
        _roi_result = (x1, y1, x2, y2)


def select_roi_interactive(frame: np.ndarray, window_name: str = "Select ROI") -> tuple:
    """
    Tampilkan frame dan minta user untuk drag-select ROI secara interaktif.
    
    Cara pakai:
    - Klik dan drag untuk memilih zona ATM
    - Tekan ENTER atau SPACE untuk konfirmasi
    - Tekan ESC atau 'r' untuk reset
    - Tekan 'q' untuk melewati (gunakan seluruh frame)
    
    Returns:
        (x1, y1, x2, y2) atau None jika dilewati
    """
    global _roi_drawing, _roi_start, _roi_end, _roi_confirmed, _roi_result
    _roi_result = None

    title = f"Pilih ROI — {window_name}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, _roi_mouse_callback)

    instruction = frame.copy()
    _add_instruction_text(instruction)

    print("\n[ROI] Klik dan drag untuk memilih zona ATM.")
    print("      ENTER/SPACE = konfirmasi | R = reset | Q = skip (seluruh frame)\n")

    while True:
        display = instruction.copy()

        # Gambar kotak sementara saat drag
        if _roi_result:
            x1, y1, x2, y2 = _roi_result
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display, f"ROI: ({x1},{y1}) → ({x2},{y2})",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif _roi_drawing:
            x1 = min(_roi_start[0], _roi_end[0])
            y1 = min(_roi_start[1], _roi_end[1])
            x2 = max(_roi_start[0], _roi_end[0])
            y2 = max(_roi_start[1], _roi_end[1])
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 1)

        cv2.imshow(title, display)
        key = cv2.waitKey(16) & 0xFF

        if key in (13, 32):  # ENTER atau SPACE
            cv2.destroyWindow(title)
            if _roi_result:
                print(f"[ROI] Dikonfirmasi: {_roi_result}")
                return _roi_result
            else:
                print("[ROI] Tidak ada ROI yang dipilih, gunakan seluruh frame.")
                return None

        elif key == ord("r"):
            _roi_result = None
            print("[ROI] Reset.")

        elif key in (ord("q"), 27):  # Q atau ESC
            cv2.destroyWindow(title)
            print("[ROI] Dilewati, gunakan seluruh frame.")
            return None


def _add_instruction_text(frame: np.ndarray):
    """Tambahkan teks instruksi ke frame."""
    h, w = frame.shape[:2]
    # Semi-transparent bar di bawah
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame,
                "Drag untuk pilih ROI  |  ENTER=konfirmasi  R=reset  Q=skip",
                (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

def setup_output_dir(path: str):
    """Buat direktori output jika belum ada."""
    Path(path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGER
# =============================================================================

def init_logger(log_path: str) -> csv.DictWriter:
    """
    Inisialisasi CSV logger untuk events loitering.
    
    Returns:
        csv.DictWriter yang sudah siap dipakai
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", newline="")
    fieldnames = [
        "video", "frame", "timestamp_sec",
        "track_id", "time_in_roi_sec", "reasons", "wall_time"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def log_event(writer: csv.DictWriter, video_name: str, event: dict):
    """Tulis satu event loitering ke CSV log."""
    writer.writerow({
        "video":           video_name,
        "frame":           event.get("frame", 0),
        "timestamp_sec":   f"{event.get('timestamp', 0):.2f}",
        "track_id":        event.get("track_id", -1),
        "time_in_roi_sec": f"{event.get('time_in_roi', 0):.1f}",
        "reasons":         event.get("reasons", ""),
        "wall_time":       time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# =============================================================================
# SCREENSHOT
# =============================================================================

def save_screenshot(frame: np.ndarray, path: str):
    """Simpan frame sebagai screenshot."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, frame)


# =============================================================================
# SUMMARY PRINTER
# =============================================================================

def print_summary(summary: dict):
    """Print ringkasan hasil processing satu video."""
    print(f"\n--- RINGKASAN: {summary.get('video', '?')} ---")
    print(f"  Total frame diproses : {summary.get('total_frames', 0)}")
    print(f"  FPS                  : {summary.get('fps', 0):.1f}")
    print(f"  Loitering terdeteksi : {'YA' if summary.get('loitering_detected') else 'TIDAK'}")
    if summary.get("loitering_ids"):
        print(f"  Track ID loitering   : {summary['loitering_ids']}")
    print(f"  Total alert events   : {summary.get('total_events', 0)}")
    print(f"  Waktu processing     : {summary.get('wall_time_sec', 0):.1f} detik")

    if summary.get("gt_has_loitering") is not None:
        gt = "YA" if summary["gt_has_loitering"] else "TIDAK"
        pred = "YA" if summary["pred_loitering"] else "TIDAK"
        match = "✓ BENAR" if summary["gt_has_loitering"] == summary["pred_loitering"] else "✗ SALAH"
        print(f"  Ground truth         : {gt}")
        print(f"  Prediksi             : {pred}  [{match}]")
    print()

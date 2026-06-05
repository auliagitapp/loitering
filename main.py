"""
main.py — Entry point sistem loitering detection.

Cara pakai:
    # Proses satu video
    python main.py --video path/to/video.mp4

    # Proses dengan ROI manual
    python main.py --video path/to/video.mp4 --roi 100 80 540 420

    # Set ROI interaktif (klik drag di frame pertama)
    python main.py --video path/to/video.mp4 --set-roi

    # Evaluasi dengan anotasi Kaggle
    python main.py --video path/to/video.mp4 --annotation annotation.csv

    # Proses semua video dalam folder
    python main.py --folder path/to/videos/ --annotation annotation.csv

    # Tanpa tampilan (processing saja)
    python main.py --video path/to/video.mp4 --no-display
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from detector import LoiteringDetector
from motion_analysis import OpticalFlowAnalyzer, PresenceHeatmap
from utils import (
    load_annotations,
    select_roi_interactive,
    setup_output_dir,
    save_screenshot,
    init_logger,
    log_event,
    print_summary,
)


def process_video(
    video_path: str,
    roi: tuple = None,
    set_roi: bool = False,
    show: bool = True,
    annotation_row: dict = None,
    output_dir: str = None,
) -> dict:
    """
    Proses satu file video dan kembalikan ringkasan hasil.

    Args:
        video_path    : path ke file video
        roi           : (x1,y1,x2,y2) zona ATM, None = seluruh frame
        set_roi       : jika True, buka frame pertama untuk pilih ROI manual
        show          : tampilkan video real-time
        annotation_row: dict dari CSV anotasi Kaggle (opsional, untuk evaluasi)
        output_dir    : direktori output, default dari config.OUTPUT_DIR

    Returns:
        dict berisi statistik hasil processing
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {video_path}")
        return {}

    # --- Info video ---
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = Path(video_path).stem

    print(f"\n{'='*60}")
    print(
        f"[VIDEO] {video_name}  |  {width}x{height}  |  {fps:.1f} fps  |  {total_frames} frames"
    )
    print(f"{'='*60}")

    # --- ROI ---
    if set_roi:
        ret, first_frame = cap.read()
        if ret:
            roi = select_roi_interactive(first_frame, video_name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
        print(f"[ROI] {roi}")
    elif roi is None:
        roi = config.ATM_ROI

    # --- Output dir ---
    out_dir = output_dir or config.OUTPUT_DIR
    vid_out_dir = os.path.join(out_dir, video_name)
    setup_output_dir(vid_out_dir)

    # --- Logger CSV ---
    log_path = os.path.join(vid_out_dir, "events.csv")
    logger = init_logger(log_path)

    # --- Video writer ---
    writer = None
    if config.SAVE_VIDEO:
        out_path = os.path.join(vid_out_dir, f"{video_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # --- Komponen utama ---
    detector = LoiteringDetector(fps=fps, roi=roi)
    flow_analyzer = OpticalFlowAnalyzer()
    heatmap = PresenceHeatmap(height, width)

    # --- Anotasi ground truth (dari Kaggle) ---
    gt_start = annotation_row.get("loiter_start", -1) if annotation_row else -1
    gt_end = annotation_row.get("loiter_end", -1) if annotation_row else -1
    has_gt = gt_start != -1

    # --- Loop utama ---
    frame_idx = 0
    total_events = 0
    start_wall = time.time()
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_time = (
            frame_idx / fps
        )  # timestamp dalam detik (deterministik untuk video file)

        # ------------------------------------------------------------------
        # 1. Deteksi + tracking + loitering logic
        # ------------------------------------------------------------------
        annotated, events = detector.process_frame(frame, current_time)

        # Update heatmap berdasarkan posisi orang yang ada di ROI
        for person in detector.track_manager.get_all_active():
            if person.currently_in_roi and person.positions:
                px, py = person.positions[-1]
                heatmap.update(int(px), int(py))

        # ------------------------------------------------------------------
        # 2. Optical flow — hitung dan overlay ke frame
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow_vis = flow_analyzer.compute_and_draw(prev_gray, gray, annotated)
            annotated = flow_vis
        prev_gray = gray

        # ------------------------------------------------------------------
        # 3. Ground truth overlay (jika ada anotasi)
        # ------------------------------------------------------------------
        if has_gt:
            annotated = _draw_gt_overlay(annotated, frame_idx, gt_start, gt_end)

        # ------------------------------------------------------------------
        # 4. Handle events (alert, screenshot, log)
        # ------------------------------------------------------------------
        for ev in events:
            total_events += 1
            log_event(logger, video_name, ev)
            print(
                f"[ALERT] {video_name} | Frame {ev['frame']} | "
                f"ID:{ev['track_id']} | {ev['reasons']} | "
                f"waktu di ROI: {ev['time_in_roi']:.1f}s"
            )
            if config.SAVE_SCREENSHOTS:
                ss_path = os.path.join(
                    vid_out_dir, f"alert_id{ev['track_id']}_f{ev['frame']}.jpg"
                )
                save_screenshot(annotated, ss_path)

        # ------------------------------------------------------------------
        # 5. Write & display
        # ------------------------------------------------------------------
        if writer:
            writer.write(annotated)

        if show and config.SHOW_VIDEO:
            cv2.imshow(f"Loitering Detection — {video_name}", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Dihentikan oleh user.")
                break
            elif key == ord("h"):
                # Tampilkan heatmap saat ini
                hm_vis = heatmap.get_visualization()
                cv2.imshow("Presence Heatmap", hm_vis)
            elif key == ord("s"):
                ss = os.path.join(vid_out_dir, f"manual_f{frame_idx}.jpg")
                save_screenshot(annotated, ss)
                print(f"[INFO] Screenshot disimpan: {ss}")

        # Progress setiap 100 frame
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_wall
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(
                f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames}  |  {elapsed:.1f}s elapsed"
            )

    # --- Selesai ---
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Simpan heatmap akhir
    hm_path = os.path.join(vid_out_dir, "presence_heatmap.jpg")
    heatmap.save(hm_path)
    print(f"[INFO] Heatmap disimpan: {hm_path}")

    # Simpan ringkasan heatmap yang di-overlay ke frame terakhir
    if "frame" in dir():  # jika ada frame yang diproses
        try:
            hm_overlay_path = os.path.join(vid_out_dir, "heatmap_overlay.jpg")
            heatmap.save_overlay(frame, hm_overlay_path)
            print(f"[INFO] Heatmap overlay disimpan: {hm_overlay_path}")
        except Exception:
            pass

    # Statistik akhir
    wall_time = time.time() - start_wall
    loitering_ids = set(
        p.track_id for p in detector.track_manager.get_loitering_persons()
    )
    loitering_ids.update(
        p.track_id
        for p in detector.track_manager.archived_tracks.values()
        if p.is_loitering
    )

    summary = {
        "video": video_name,
        "total_frames": frame_idx,
        "fps": fps,
        "loitering_detected": len(loitering_ids) > 0,
        "loitering_ids": list(loitering_ids),
        "total_events": total_events,
        "wall_time_sec": wall_time,
        # "gt_has_loitering": has_gt and gt_start != -1,
        "gt_has_loitering": (gt_start != -1) if annotation_row else None,
        "pred_loitering": len(loitering_ids) > 0,
    }

    print_summary(summary)
    return summary


def _draw_gt_overlay(
    frame: np.ndarray, frame_idx: int, gt_start: int, gt_end: int
) -> np.ndarray:
    """Gambar indikator ground truth di frame (untuk evaluasi)."""
    h, w = frame.shape[:2]

    # Bar progress GT di bagian bawah
    bar_y = h - 18
    bar_h = 6

    # Background bar
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)

    # Zona loitering GT
    if gt_start > 0 and gt_end > 0:
        total = max(gt_end, frame_idx)
        x1 = int(gt_start / total * w)
        x2 = int(gt_end / total * w)
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (0, 80, 255), -1)

    # Posisi frame saat ini
    if gt_end > 0:
        cur_x = int(frame_idx / gt_end * w)
        cv2.line(
            frame, (cur_x, bar_y - 2), (cur_x, bar_y + bar_h + 2), (255, 255, 255), 1
        )

    # Label GT
    in_gt_zone = gt_start <= frame_idx <= gt_end if gt_start > 0 else False
    label = "GT:LOITERING" if in_gt_zone else "GT:normal"
    color = (0, 80, 255) if in_gt_zone else (120, 120, 120)
    cv2.putText(frame, label, (6, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="ATM Loitering Detection System")

    parser.add_argument("--video", type=str, help="Path ke file video")
    parser.add_argument("--folder", type=str, help="Folder berisi banyak video")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Koordinat ROI: x1 y1 x2 y2",
    )
    parser.add_argument(
        "--set-roi",
        action="store_true",
        help="Pilih ROI secara interaktif dari frame pertama",
    )
    parser.add_argument(
        "--annotation", type=str, help="Path ke CSV anotasi Kaggle untuk evaluasi"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Jangan tampilkan video (lebih cepat)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.OUTPUT_DIR,
        help=f"Direktori output (default: {config.OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Validasi input
    if not args.video and not args.folder:
        parser.error("Berikan --video atau --folder")

    roi = tuple(args.roi) if args.roi else None
    show = not args.no_display

    # Load anotasi jika ada
    annotations = {}
    if args.annotation:
        annotations = load_annotations(args.annotation)
        print(f"[INFO] Loaded {len(annotations)} anotasi dari {args.annotation}")

    # Kumpulkan video yang akan diproses
    video_files = []
    if args.video:
        video_files = [args.video]
    elif args.folder:
        exts = {".mp4", ".avi", ".mkv", ".mov", ".MP4"}
        folder = Path(args.folder)
        video_files = [str(f) for f in folder.iterdir() if f.suffix in exts]
        video_files.sort()
        print(f"[INFO] Ditemukan {len(video_files)} video di {args.folder}")

    # Proses semua video
    all_summaries = []
    for vp in video_files:
        vname = Path(vp).name
        ann_row = annotations.get(vname)
        summary = process_video(
            video_path=vp,
            roi=roi,
            set_roi=args.set_roi
            and len(video_files) == 1,  # set-roi hanya untuk 1 video
            show=show,
            annotation_row=ann_row,
            output_dir=args.output,
        )
        all_summaries.append(summary)

    # Evaluasi keseluruhan jika ada GT
    if annotations and len(all_summaries) > 1:
        _print_evaluation(all_summaries)


def _print_evaluation(summaries: list):
    """Hitung precision/recall sederhana dari semua video."""
    print(f"\n{'='*60}")
    print("EVALUASI KESELURUHAN")
    print(f"{'='*60}")

    tp = fp = tn = fn = 0
    for s in summaries:
        pred = s.get("pred_loitering", False)
        gt = s.get("gt_has_loitering", False)
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"  Total video : {total}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision   : {precision:.3f}")
    print(f"  Recall      : {recall:.3f}")
    print(f"  F1 Score    : {f1:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

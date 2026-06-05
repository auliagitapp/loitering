"""
detector.py — Integrasi YOLOv8 + ByteTrack + visualisasi alert loitering.

LOITERING : kotak merah + label "! LOITERING [alasan]"
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import config
from tracker import TrackManager, PersonState


def is_inside_roi(cx: float, cy: float, roi: tuple) -> bool:
    if roi is None:
        return True
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


class LoiteringDetector:

    def __init__(self, fps: float, roi: tuple = None):
        print(f"[INFO] Loading YOLOv8 model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)

        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=config.YOLO_CONFIDENCE,
                lost_track_buffer=config.TRACK_LOST_FRAMES,
                minimum_matching_threshold=config.TRACK_IOU_THRESHOLD,
                frame_rate=int(fps),
            )
        except AttributeError:
            self.tracker = sv.ByteTracker(
                track_activation_threshold=config.YOLO_CONFIDENCE,
                lost_track_buffer=config.TRACK_LOST_FRAMES,
                minimum_matching_threshold=config.TRACK_IOU_THRESHOLD,
                frame_rate=int(fps),
            )

        self.fps = fps
        self.roi = roi
        self.track_manager = TrackManager(fps)
        self.frame_count = 0

        print(f"[INFO] Detector siap. ROI: {roi if roi else 'seluruh frame'}")

    def set_roi(self, roi: tuple):
        self.roi = roi

    def process_frame(self, frame: np.ndarray, current_time: float) -> tuple:
        self.frame_count += 1

        # ------------------------------------------------------------------
        # YOLOv8 deteksi person
        # ------------------------------------------------------------------
        results = self.model(
            frame, verbose=False,
            conf=config.YOLO_CONFIDENCE,
            classes=[0]
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        tracked = self.tracker.update_with_detections(detections)

        # ------------------------------------------------------------------
        # Cek ROI per orang
        # ------------------------------------------------------------------
        track_list = []
        roi_flags  = []

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            bbox = (x1, y1, x2, y2)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            track_list.append((track_id, bbox))
            roi_flags.append(is_inside_roi(cx, cy, self.roi))

        # ------------------------------------------------------------------
        # Update TrackManager
        # ------------------------------------------------------------------
        self.track_manager.update_tracks(
            track_list, roi_flags, current_time
        )

        # ------------------------------------------------------------------
        # Annotasi frame
        # ------------------------------------------------------------------
        annotated = self._annotate_frame(frame.copy(), track_list, roi_flags)

        # ------------------------------------------------------------------
        # Kumpulkan events
        # ------------------------------------------------------------------
        events = self._collect_events(current_time)

        return annotated, events

    def _annotate_frame(self, frame, track_list, roi_flags):
        h, w = frame.shape[:2]

        # Garis ROI
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), config.COLOR_ROI, 2)
            cv2.putText(frame, "ATM ZONE", (rx1 + 4, ry1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_ROI, 1)

        for (track_id, bbox), in_roi in zip(track_list, roi_flags):
            person = self.track_manager.active_tracks.get(track_id)
            if person is None:
                continue

            x1, y1, x2, y2 = bbox
            color = person.get_status_color()

            thickness = 3 if person.is_loitering else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label ID + waktu
            time_str = f"{person.time_in_roi:.1f}s"
            label = f"ID:{track_id} {time_str}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Label alert di bawah bbox
            if person.is_loitering:
                reason = person.flags.describe()
                alert_label = f"! LOITERING {reason}"

                font_scale = 0.6
                (aw, ah), _ = cv2.getTextSize(
                    alert_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

                cv2.rectangle(frame, (x1, y2), (x1 + aw + 8, y2 + ah + 10),
                              color, -1)
                cv2.putText(frame, alert_label, (x1 + 4, y2 + ah + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), 2)

            # Trajectory
            if len(person.positions) > 1:
                pts = list(person.positions)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(ch * alpha) for ch in color)
                    cv2.line(frame,
                             (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])),
                             c, 1)

        # Info ringkas
        # Hitung berdasarkan PersonState aktif saat ini saja (bukan akumulasi events)
        # Ini mencegah "Loitering: 28" padahal orangnya 1
        all_persons = self.track_manager.get_all_active()
        loitering_persons = [p for p in all_persons if p.is_loitering]
        total_count       = len(track_list)

        # Tampilkan jumlah orang SAAT INI, bukan total akumulasi
        info = f"People: {total_count}"
        if loitering_persons:
            info += f"   ! LOITERING"

        info_color = config.COLOR_LOITERING if loitering_persons else (0, 200, 0)

        cv2.putText(frame, info, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, info, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 1)

        cv2.putText(frame, f"Frame: {self.frame_count}",
                    (w - 130, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return frame

    def _collect_events(self, current_time: float) -> list:
        events = []
        for person in self.track_manager.get_loitering_persons():
            if self.track_manager.should_send_alert(person):
                events.append({
                    "track_id":    person.track_id,
                    "time_in_roi": person.time_in_roi,
                    "reasons":     person.flags.describe(),
                    "level":       "LOITERING",
                    "timestamp":   current_time,
                    "frame":       self.frame_count,
                })
        return events

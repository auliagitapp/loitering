"""
motion_analysis.py — Bukti visual pergerakan: optical flow + presence heatmap.

Dua komponen utama:
  1. OpticalFlowAnalyzer  — Farneback dense optical flow, overlay panah ke frame
  2. PresenceHeatmap      — Akumulasi posisi orang menjadi heatmap panas/dingin

Mengapa ini penting sebagai "bukti":
  - Optical flow membuktikan ADA atau TIDAK ADA gerakan di zona ATM
    bahkan jika orang berdiri diam (gerakan tangan/tubuh halus tetap terdeteksi)
  - Heatmap membuktikan AREA MANA yang paling sering ditempati —
    orang yang loitering akan meninggalkan "titik panas" yang konsisten
"""

import cv2
import numpy as np
from pathlib import Path


class OpticalFlowAnalyzer:
    """
    Dense optical flow menggunakan algoritma Farneback.
    
    Farneback dipilih karena:
    - Tidak butuh GPU
    - Memberikan flow di SETIAP piksel (dense), bukan hanya titik tertentu
    - Cocok untuk mendeteksi gerakan halus (tangan di ATM keypad)
    
    Output visual: panah-panah kecil yang menunjukkan arah dan besar gerakan.
    Hanya piksel dengan magnitude > threshold yang ditampilkan agar tidak crowded.
    """

    def __init__(
        self,
        sampling_step: int = 16,        # jarak antar panah (piksel)
        arrow_scale: float = 3.0,       # panjang panah relatif terhadap magnitude
        magnitude_threshold: float = 1.5,  # abaikan gerakan sangat kecil (noise)
        arrow_color: tuple = (0, 255, 255),  # kuning-cyan
        arrow_thickness: int = 1,
        alpha: float = 0.5,             # transparansi overlay panah
    ):
        self.step      = sampling_step
        self.scale     = arrow_scale
        self.threshold = magnitude_threshold
        self.color     = arrow_color
        self.thickness = arrow_thickness
        self.alpha     = alpha

        # Parameter Farneback — tidak perlu diubah untuk kebanyakan kasus
        self._fb_params = dict(
            pyr_scale=0.5,      # skala pyramid (0.5 = standard)
            levels=3,           # jumlah level pyramid
            winsize=15,         # ukuran window averaging
            iterations=3,       # iterasi per level
            poly_n=5,           # ukuran pixel neighborhood
            poly_sigma=1.2,     # sigma Gaussian untuk polynomial
            flags=0,
        )

        # Akumulasi magnitude untuk statistik
        self._frame_count = 0
        self._total_magnitude = 0.0

    def compute_and_draw(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Hitung optical flow antara dua frame grayscale dan overlay ke frame BGR.
        
        Args:
            prev_gray : frame sebelumnya (grayscale)
            curr_gray : frame saat ini (grayscale)
            frame_bgr : frame saat ini (BGR) — akan diberi overlay panah
            
        Returns:
            Frame dengan overlay panah optical flow
        """
        # Hitung dense flow: setiap piksel punya vektor (fx, fy)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self._fb_params
        )

        # Konversi flow ke magnitude dan angle
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx**2 + fy**2)

        # Update statistik
        self._frame_count += 1
        self._total_magnitude += float(magnitude.mean())

        # Buat layer overlay transparan
        overlay = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        # Gambar panah hanya di sampling grid
        for y in range(0, h, self.step):
            for x in range(0, w, self.step):
                mag = magnitude[y, x]
                if mag < self.threshold:
                    continue  # abaikan noise / piksel diam

                # Endpoint panah
                end_x = int(x + fx[y, x] * self.scale)
                end_y = int(y + fy[y, x] * self.scale)

                # Warna berdasarkan magnitude: biru (lambat) → merah (cepat)
                norm_mag = min(mag / 10.0, 1.0)
                color = (
                    int(255 * (1 - norm_mag)),  # B
                    int(100 * (1 - norm_mag)),  # G
                    int(255 * norm_mag),         # R
                )

                cv2.arrowedLine(
                    overlay,
                    (x, y), (end_x, end_y),
                    color, self.thickness,
                    tipLength=0.3,
                )

        # Blend overlay dengan frame asli
        result = cv2.addWeighted(overlay, self.alpha, frame_bgr, 1 - self.alpha, 0)

        # Tambahkan indikator magnitude rata-rata di frame
        avg_mag = self._total_magnitude / self._frame_count
        self._draw_motion_indicator(result, float(magnitude.mean()), avg_mag)

        return result

    def _draw_motion_indicator(
        self,
        frame: np.ndarray,
        current_mag: float,
        avg_mag: float,
    ):
        """
        Gambar indikator magnitude motion di pojok kanan atas.
        Bar pendek = sedikit gerakan, bar panjang = banyak gerakan.
        """
        h, w = frame.shape[:2]
        bar_x = w - 160
        bar_y = 10
        bar_w = 140
        bar_h = 14

        # Background
        cv2.rectangle(frame, (bar_x - 4, bar_y - 2),
                      (bar_x + bar_w + 4, bar_y + bar_h + 16), (0, 0, 0), -1)

        # Bar motion saat ini
        fill = min(int(current_mag / 8.0 * bar_w), bar_w)
        color = (0, 255, 0) if current_mag < 2.0 else \
                (0, 165, 255) if current_mag < 5.0 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), 1)

        # Label
        cv2.putText(frame, f"Motion: {current_mag:.2f}",
                    (bar_x, bar_y + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    def get_flow_mask(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi: tuple = None,
    ) -> np.ndarray:
        """
        Kembalikan binary mask area yang bergerak (untuk analisis lanjutan).
        Berguna untuk mendeteksi gerakan tangan di sekitar keypad ATM.
        """
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self._fb_params
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mask = (mag > self.threshold).astype(np.uint8) * 255

        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_mask = np.zeros_like(mask)
            roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            return roi_mask

        return mask

    @property
    def average_magnitude(self) -> float:
        """Rata-rata magnitude motion selama video."""
        if self._frame_count == 0:
            return 0.0
        return self._total_magnitude / self._frame_count


class PresenceHeatmap:
    """
    Akumulasi posisi orang menjadi heatmap untuk bukti visual loitering.
    
    Cara kerja:
    - Setiap frame, setiap posisi orang di ROI menambah nilai ke heatmap
    - Area yang sering ditempati = nilai tinggi = warna merah
    - Area yang jarang/tidak pernah ditempati = nilai rendah = warna biru/hitam
    
    Output: gambar heatmap COLORMAP_JET (biru→hijau→kuning→merah)
    Merah = orang paling sering di sana = bukti loitering.
    """

    def __init__(
        self,
        height: int,
        width: int,
        blur_radius: int = 25,    # radius Gaussian blur untuk smoothing
        decay: float = 0.998,     # faktor peluruhan agar heatmap tidak terlalu saturated
    ):
        self.height = height
        self.width  = width
        self.blur   = blur_radius if blur_radius % 2 == 1 else blur_radius + 1  # harus ganjil
        self.decay  = decay

        # Accumulator — float32 agar tidak overflow
        self._map = np.zeros((height, width), dtype=np.float32)

        # Statistik
        self._update_count = 0

    def update(self, px: int, py: int, weight: float = 1.0):
        """
        Tambahkan kehadiran orang di posisi (px, py).
        
        Args:
            px, py : koordinat piksel (center bounding box)
            weight : bobot kontribusi (1.0 = normal)
        """
        if 0 <= px < self.width and 0 <= py < self.height:
            # Tambah nilai di posisi orang
            self._map[py, px] += weight
            self._update_count += 1

        # Peluruhan: kurangi sedikit semua nilai agar momen lama fade
        # Di-apply hanya setiap 30 update agar tidak terlalu sering (efisiensi)
        if self._update_count % 30 == 0:
            self._map *= self.decay

    def update_bbox(self, x1: int, y1: int, x2: int, y2: int, weight: float = 0.5):
        """
        Tambahkan kehadiran orang berdasarkan seluruh bounding box area.
        Lebih akurat dari hanya satu titik center.
        """
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.width - 1, x2)
        y2 = min(self.height - 1, y2)

        if x2 > x1 and y2 > y1:
            self._map[y1:y2, x1:x2] += weight * 0.1  # spread tipis di seluruh bbox
            # Titik paling berat di center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r = min((x2 - x1) // 4, (y2 - y1) // 4, 20)
            cv2.circle(self._map, (cx, cy), r, weight, -1)

    def get_visualization(self, normalize: bool = True) -> np.ndarray:
        """
        Kembalikan visualisasi heatmap sebagai gambar BGR.
        
        Args:
            normalize: jika True, normalisasi ke 0-255 sebelum colormap
            
        Returns:
            BGR image ukuran (height, width)
        """
        # Gaussian blur untuk smoothing
        blurred = cv2.GaussianBlur(self._map, (self.blur, self.blur), 0)

        # Normalisasi
        if normalize and blurred.max() > 0:
            norm = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            norm = np.clip(blurred, 0, 255).astype(np.uint8)

        # Terapkan colormap JET: biru (dingin) → merah (panas)
        heatmap_colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        return heatmap_colored

    def get_overlay(self, background: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap di atas gambar background (biasanya frame video).
        
        Args:
            background : frame BGR ukuran sama dengan heatmap
            alpha      : transparansi heatmap (0=tidak terlihat, 1=penuh)
            
        Returns:
            Frame dengan heatmap overlay
        """
        if background.shape[:2] != (self.height, self.width):
            background = cv2.resize(background, (self.width, self.height))

        hm = self.get_visualization()

        # Hanya overlay area yang punya nilai > threshold (hindari overlay area kosong)
        blurred = cv2.GaussianBlur(self._map, (self.blur, self.blur), 0)
        if blurred.max() > 0:
            significance = (blurred / blurred.max())
        else:
            significance = blurred

        # Buat alpha mask berdasarkan signifikansi
        alpha_mask = (significance * alpha).clip(0, alpha)
        alpha_3ch = np.stack([alpha_mask] * 3, axis=-1).astype(np.float32)

        result = (
            background.astype(np.float32) * (1 - alpha_3ch)
            + hm.astype(np.float32) * alpha_3ch
        ).clip(0, 255).astype(np.uint8)

        # Tambahkan colorbar legend
        result = self._add_colorbar(result)

        return result

    def _add_colorbar(self, frame: np.ndarray) -> np.ndarray:
        """Tambahkan colorbar vertikal di kanan frame sebagai legenda."""
        h, w = frame.shape[:2]
        bar_w = 14
        bar_x = w - bar_w - 8
        bar_y1 = 20
        bar_y2 = h - 20

        # Gambar gradient bar (merah atas = banyak, biru bawah = sedikit)
        for y in range(bar_y1, bar_y2):
            ratio = 1.0 - (y - bar_y1) / (bar_y2 - bar_y1)
            val = int(ratio * 255)
            color_strip = np.zeros((1, 1), dtype=np.uint8)
            color_strip[0, 0] = val
            colored = cv2.applyColorMap(color_strip, cv2.COLORMAP_JET)[0, 0].tolist()
            cv2.line(frame, (bar_x, y), (bar_x + bar_w, y), tuple(colored), 1)

        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + bar_w, bar_y2), (200, 200, 200), 1)
        cv2.putText(frame, "High", (bar_x - 2, bar_y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, "Low", (bar_x - 2, bar_y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return frame

    def save(self, path: str):
        """Simpan visualisasi heatmap ke file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        hm = self.get_visualization()
        cv2.imwrite(path, hm)
        print(f"[INFO] Heatmap disimpan: {path}")

    def save_overlay(self, background: np.ndarray, path: str, alpha: float = 0.55):
        """Simpan overlay heatmap + background ke file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        overlay = self.get_overlay(background, alpha)

        # Tambahkan judul
        cv2.putText(overlay, "Presence Heatmap — ATM Loitering Evidence",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(path, overlay)

    def reset(self):
        """Reset heatmap (untuk mulai video baru)."""
        self._map = np.zeros((self.height, self.width), dtype=np.float32)
        self._update_count = 0

    @property
    def max_value(self) -> float:
        """Nilai maksimum di heatmap (indikasi intensitas loitering)."""
        return float(self._map.max())

    @property
    def hot_area_ratio(self) -> float:
        """Proporsi area yang tergolong 'panas' (> 50% dari max)."""
        if self._map.max() == 0:
            return 0.0
        hot_pixels = (self._map > self._map.max() * 0.5).sum()
        total_pixels = self.height * self.width
        return float(hot_pixels / total_pixels)

#!/usr/bin/env python3
"""
radar_visualizer.py — SENTINEL Mesh Radar Console
=================================================
A two-mode range-Doppler playback console for the integrated RD-video
datasets produced by ``integrate_simulations.py``.

Modes (toggle with TAB or the on-screen button):

* TECHNICAL  — full engineering view: range-Doppler heatmap with a dB
  colour scale, live CFAR detections, the ground-truth target cell,
  confirmed Kalman tracks, the picket-fence geometry, and the complete
  per-frame telemetry + radar configuration.
* PRESENTATION — a clean, self-explaining briefing view for a
  non-technical audience: a plain-language target card, an honest
  detection-status banner, and the picket-fence engagement profile with
  a live "detecting here" marker.

Detection status is **truth-gated**: the banner reads TARGET TRACKED
only when the CFAR detector actually fires within a few cells of the
known target location for the current frame.  Nothing is hard-coded.

Controls
--------
    SPACE        play / pause
    ← / →        step one frame
    TAB          switch TECHNICAL / PRESENTATION
    [ / ]        previous / next dataset
    D            toggle CFAR detection overlay   (technical)
    T            toggle track overlay            (technical)
    C            cycle CFAR algorithm CA / OS    (technical)
    ESC / Q      quit
Click the timeline to scrub; click the mode / play buttons.

Usage
-----
    python radar_visualizer.py                       # integrated_output/
    python radar_visualizer.py --data_dir some_dir
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndimage
import pygame


# ═══════════════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════════════

BG        = (9, 12, 18)          # app background
PANEL     = (16, 21, 30)         # panel fill
PANEL_HI  = (21, 28, 40)         # raised element
BORDER    = (38, 50, 68)         # panel border
GRID      = (28, 38, 52)         # heatmap gridlines

TEXT      = (224, 232, 240)      # primary text
MUTED     = (120, 140, 162)      # labels / secondary
FAINT     = (70, 86, 104)        # tertiary

CYAN      = (0, 200, 255)        # presentation accent
GREEN     = (38, 222, 130)       # confirmed / good
AMBER     = (255, 184, 28)       # caution / searching
RED       = (255, 74, 92)        # threat / target truth
WHITE     = (245, 250, 255)

THREAT_COL = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}


# ═══════════════════════════════════════════════════════════════════════════
# Signal-processing backend (CFAR, detection clustering, tracking)
# ═══════════════════════════════════════════════════════════════════════════

class VectorizedCFAR:
    """Vectorized cell-averaging / ordered-statistic CFAR (Doppler axis)."""

    @staticmethod
    def detect_ca_cfar(power_linear, guard=2, training=8, pfa=1e-4):
        win_total = 2 * (guard + training) + 1
        win_guard = 2 * guard + 1
        sum_total = ndimage.uniform_filter1d(
            power_linear, size=win_total, axis=1, mode="wrap") * win_total
        sum_guard = ndimage.uniform_filter1d(
            power_linear, size=win_guard, axis=1, mode="wrap") * win_guard
        n = 2 * training
        noise_est = (sum_total - sum_guard) / n
        alpha = n * (pfa ** (-1.0 / n) - 1.0)
        return (power_linear > noise_est * alpha).astype(np.int32)

    @staticmethod
    def detect_os_cfar(power_linear, guard=2, training=8, pfa=1e-4, k_ratio=0.75):
        half_w = guard + training
        padded = np.pad(power_linear, ((0, 0), (half_w, half_w)), mode="wrap")
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(
            padded, (1, 2 * half_w + 1), axis=(0, 1)).squeeze(axis=2)
        left = windows[:, :, :training]
        right = windows[:, :, training + 2 * guard + 1:]
        train = np.concatenate([left, right], axis=2)
        n = 2 * training
        k = min(max(int(round(k_ratio * n)), 1), n)
        os_vals = np.partition(train, k - 1, axis=2)[:, :, k - 1]
        alpha = (pfa ** (-1.0 / (n - k + 1))) - 1.0
        return (power_linear > os_vals * alpha).astype(np.int32)


def extract_detections_centroids(cfar_map, range_resolution_m, prf_hz,
                                 cpi_pulses, lambda_m, bistatic_angle_deg):
    """Group active CFAR pixels into target cluster centroids."""
    active_rows, active_cols = np.where(cfar_map > 0)
    if len(active_rows) == 0:
        return []

    visited = set()
    detections = []
    cos_half = math.cos(math.radians(bistatic_angle_deg / 2.0))

    for i in range(len(active_rows)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        queue = [i]
        while queue:
            curr = queue.pop(0)
            r1, c1 = active_rows[curr], active_cols[curr]
            for j in range(len(active_rows)):
                if j in visited:
                    continue
                r2, c2 = active_rows[j], active_cols[j]
                dc = min(abs(c1 - c2), cpi_pulses - abs(c1 - c2))
                if abs(r1 - r2) <= 5 and dc <= 4:
                    cluster.append(j)
                    visited.add(j)
                    queue.append(j)

        cluster_rows = active_rows[cluster]
        cluster_cols = active_cols[cluster]
        mean_r_bin = float(np.mean(cluster_rows))
        angles = cluster_cols * (2.0 * np.pi / cpi_pulses)
        mean_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        mean_c_bin = float((mean_angle / (2.0 * np.pi) * cpi_pulses) % cpi_pulses)

        range_m = mean_r_bin * range_resolution_m
        fd_res = prf_hz / cpi_pulses
        bin_idx = int(round(mean_c_bin)) % cpi_pulses
        fd_aliased = bin_idx * fd_res
        if fd_aliased >= prf_hz / 2.0:
            fd_aliased -= prf_hz
        vr_mps = fd_aliased * lambda_m / (2.0 * cos_half)

        detections.append({
            "range_m": range_m, "vr_mps": vr_mps,
            "r_bin": mean_r_bin, "d_bin": mean_c_bin, "size": len(cluster),
        })
    return detections


class TargetTrack:
    """Single persistent target track maintained by a 2-state Kalman filter."""

    def __init__(self, track_id, init_r, init_vr):
        self.track_id = track_id
        self.x = np.array([[init_r], [init_vr]], dtype=float)
        self.P = np.array([[100.0, 0.0], [0.0, 25.0]], dtype=float)
        self.state = "TENTATIVE"
        self.history: list[tuple[float, float]] = []
        self.hit_count = 1
        self.miss_count = 0
        self.age = 1

    def predict(self, dt, sigma_a=3.0):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([
            [0.25 * dt**4 * sigma_a**2, 0.5 * dt**3 * sigma_a**2],
            [0.5 * dt**3 * sigma_a**2, dt**2 * sigma_a**2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1

    def update(self, z_r, z_vr_unwrapped, r_noise=3.0, vr_noise=1.0):
        z = np.array([[z_r], [z_vr_unwrapped]])
        H = np.eye(2)
        R = np.diag([r_noise**2, vr_noise**2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(2) - K @ H) @ self.P
        self.hit_count += 1
        self.miss_count = 0
        self.history.append((float(self.x[0, 0]), float(self.x[1, 0])))
        if len(self.history) > 60:
            self.history.pop(0)


class MultiTargetTracker:
    """Multi-target tracking with gated association and Doppler unwrapping."""

    def __init__(self, prf_hz, lambda_m, bistatic_angle_deg):
        self.tracks: list[TargetTrack] = []
        self.next_track_id = 1
        self.prf_hz = prf_hz
        self.lambda_m = lambda_m
        self.bistatic_angle_deg = bistatic_angle_deg
        cos_half = math.cos(math.radians(bistatic_angle_deg / 2.0))
        self.v_ambig = prf_hz * lambda_m / (2.0 * cos_half)

    def process_frame(self, detections, dt):
        for track in self.tracks:
            track.predict(dt)

        gate_r, gate_v = 150.0, 40.0
        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()

        for t_idx, track in enumerate(self.tracks):
            pred_r, pred_vr = track.x[0, 0], track.x[1, 0]
            best_idx, best_dist, best_vr = -1, float("inf"), 0.0
            for d_idx, det in enumerate(detections):
                if d_idx in matched_detections:
                    continue
                det_r = det["range_m"]
                # Unwrap to the PRF fold nearest the prediction (O(1)).
                n = round((pred_vr - det["vr_mps"]) / self.v_ambig)
                det_vr = det["vr_mps"] + n * self.v_ambig
                dist = math.sqrt(((det_r - pred_r) / gate_r)**2
                                 + ((det_vr - pred_vr) / gate_v)**2)
                if dist < 1.0 and dist < best_dist:
                    best_dist, best_idx, best_vr = dist, d_idx, det_vr
            if best_idx != -1:
                matched_tracks.add(t_idx)
                matched_detections.add(best_idx)
                track.update(detections[best_idx]["range_m"], best_vr)

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.miss_count += 1
                track.P *= 1.1
                track.history.append((float(track.x[0, 0]), float(track.x[1, 0])))
                if len(track.history) > 60:
                    track.history.pop(0)

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                init_vr_aliased = det["vr_mps"]
                n = 258 if init_vr_aliased >= 0 else -258
                init_vr = init_vr_aliased + n * self.v_ambig
                self.tracks.append(
                    TargetTrack(self.next_track_id, det["range_m"], init_vr))
                self.next_track_id += 1

        active = []
        for track in self.tracks:
            if track.state == "TENTATIVE" and track.hit_count >= 3:
                track.state = "CONFIRMED"
            if track.state == "CONFIRMED" and track.miss_count < 6:
                active.append(track)
            elif track.state == "TENTATIVE" and track.miss_count < 2:
                active.append(track)
        self.tracks = active


def _clip(v):
    return max(0.0, min(255.0, v))


def make_lut(palette):
    """Build a 256-level RGB lookup table. palette: 'green' or 'cyan'."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        if palette == "green":
            r = _clip((v ** 4.5) * 220)
            g = _clip(v * 215 + (v ** 2) * 40 if v > 0.05 else v * 150)
            b = _clip((v ** 2.2) * 165 + (v ** 6) * 90 if v > 0.1 else v * 80)
        else:  # cyan
            r = _clip((v ** 4) * 150)
            g = _clip((v ** 1.8) * 190 + (v ** 4) * 55)
            b = _clip(v * 235 + (v ** 0.5) * 20)
        lut[i] = [r, g, b]
    return lut


# ═══════════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════════

def font(size, bold=False):
    return pygame.font.SysFont(
        "Helvetica Neue,Helvetica,Arial,DejaVu Sans", size, bold=bold)


class Console:
    """The two-mode radar playback console."""

    W, H = 1440, 880

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.files = sorted(data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz datasets in {data_dir.resolve()}")
        self.file_idx = 0

        pygame.init()
        pygame.display.set_caption("SENTINEL Mesh — Radar Console")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()

        self.f_h1 = font(30, bold=True)
        self.f_h2 = font(20, bold=True)
        self.f_lbl = font(15, bold=True)
        self.f_txt = font(16)
        self.f_sm = font(13)
        self.f_big = font(46, bold=True)
        self.f_mid = font(26, bold=True)

        self.mode = "presentation"      # or "technical"
        self.playing = True
        self.speed = 1.0
        self.last_tick = pygame.time.get_ticks() / 1000.0

        self.show_detections = True
        self.show_tracks = True
        self.cfar_algo = "CA-CFAR"
        self.pfa = 1e-4
        self.db_min, self.db_max = 8.0, 50.0

        self.lut_green = make_lut("green")
        self.lut_cyan = make_lut("cyan")

        self._buttons: dict[str, pygame.Rect] = {}
        self._load(self.files[self.file_idx])

    # ── data ──────────────────────────────────────────────────────────

    def _load(self, path: Path):
        self.path = path
        data = np.load(path, allow_pickle=True)
        self.rd = data["rd_video"]
        self.meta = data["metadata"].tolist()
        self.cfg = data["radar_config"].tolist()
        self.n_frames, self.n_range, self.n_dop = self.rd.shape
        self.frame = 0

        self.prf = float(self.cfg["prf_hz"])
        self.cpi = int(self.cfg["cpi_pulses"])
        self.lam = float(self.cfg["lambda_m"])
        self.beta = float(self.cfg["bistatic_angle_deg"])
        self.range_res = float(self.cfg["range_resolution_m"])
        self.drone_alt = float(self.cfg["drone_pos"][2])
        self.dt = (float(self.meta[1]["t_s"] - self.meta[0]["t_s"])
                   if self.n_frames > 1 else 0.128)
        self.int_gain_db = 10.0 * math.log10(self.cpi)

        self._precompute()
        self._eval_frame()

    def _precompute(self):
        """One-time per-dataset pass: CFAR + clustering + tracking per frame,
        cached so frame stepping/playback is O(1)."""
        self.cache: list[dict] = []
        tracker = MultiTargetTracker(self.prf, self.lam, self.beta)
        cfar = (VectorizedCFAR.detect_os_cfar if self.cfar_algo == "OS-CFAR"
                else VectorizedCFAR.detect_ca_cfar)
        for f in range(self.n_frames):
            power_lin = 10.0 ** (self.rd[f] / 10.0)
            det_map = cfar(power_lin, pfa=self.pfa)
            dets = extract_detections_centroids(
                det_map, self.range_res, self.prf, self.cpi, self.lam, self.beta)
            tracker.process_frame(dets, self.dt)

            m = self.meta[f]
            truth_in = bool(m["target_in_bounds"])
            truth_detected = False
            truth_rd = None
            if truth_in:
                r_bin = int(m["range_bin"])
                d_shift = (int(m["doppler_bin"]) + self.n_dop // 2) % self.n_dop
                truth_rd = (r_bin, d_shift)
                lo_r, hi_r = max(0, r_bin - 3), min(self.n_range, r_bin + 4)
                d_idx = [(d_shift + k) % self.n_dop for k in range(-3, 4)]
                truth_detected = bool(det_map[lo_r:hi_r][:, d_idx].any())

            self.cache.append({
                "dets": dets,
                "n_dets": int(det_map.sum()),
                "truth_in": truth_in,
                "truth_detected": truth_detected,
                "truth_rd": truth_rd,
                "snapshot": [
                    {"id": t.track_id, "x": t.x.copy(), "P": t.P.copy(),
                     "state": t.state, "miss": t.miss_count}
                    for t in tracker.tracks
                ],
            })

    def _eval_frame(self):
        """Pull cached per-frame detection state for the current frame (O(1))."""
        c = self.cache[self.frame]
        m = self.meta[self.frame]
        self.dets = c["dets"]
        self.n_dets = c["n_dets"]
        self.truth_in = c["truth_in"]
        self.truth_detected = c["truth_detected"]
        self.truth_rd = c["truth_rd"]
        self.snapshot = c["snapshot"]
        self.snr_eff_db = float(m["snr_single_pulse_db"]) + self.int_gain_db

    # ── playback / events ─────────────────────────────────────────────

    def run(self):
        running = True
        while running:
            running = self._events()
            self._advance()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def _advance(self):
        if not self.playing or self.n_frames <= 1:
            return
        now = pygame.time.get_ticks() / 1000.0
        if now - self.last_tick >= self.dt / max(self.speed, 1e-3):
            self.frame = (self.frame + 1) % self.n_frames
            self.last_tick = now
            self._eval_frame()

    def _set_frame(self, f):
        self.frame = max(0, min(self.n_frames - 1, f))
        self._eval_frame()

    def _cycle(self, off):
        self.file_idx = (self.file_idx + off) % len(self.files)
        self._load(self.files[self.file_idx])

    def _events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif e.key == pygame.K_SPACE:
                    self.playing = not self.playing
                elif e.key == pygame.K_TAB:
                    self.mode = "technical" if self.mode == "presentation" else "presentation"
                elif e.key == pygame.K_RIGHT:
                    self.playing = False; self._set_frame(self.frame + 1)
                elif e.key == pygame.K_LEFT:
                    self.playing = False; self._set_frame(self.frame - 1)
                elif e.key == pygame.K_RIGHTBRACKET:
                    self._cycle(1)
                elif e.key == pygame.K_LEFTBRACKET:
                    self._cycle(-1)
                elif e.key == pygame.K_d:
                    self.show_detections = not self.show_detections
                elif e.key == pygame.K_t:
                    self.show_tracks = not self.show_tracks
                elif e.key == pygame.K_c:
                    self.cfar_algo = "OS-CFAR" if self.cfar_algo == "CA-CFAR" else "CA-CFAR"
                    self._precompute()
                    self._eval_frame()
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._click(e.pos)
        return True

    def _click(self, pos):
        for name, rect in self._buttons.items():
            if rect.collidepoint(pos):
                if name == "mode":
                    self.mode = "technical" if self.mode == "presentation" else "presentation"
                elif name == "play":
                    self.playing = not self.playing
                elif name == "prev":
                    self._cycle(-1)
                elif name == "next":
                    self._cycle(1)
                elif name.startswith("run:"):
                    self.file_idx = int(name.split(":")[1])
                    self._load(self.files[self.file_idx])
                elif name == "timeline":
                    rel = (pos[0] - rect.x) / max(rect.width, 1)
                    self.playing = False
                    self._set_frame(int(rel * (self.n_frames - 1)))
                return

    # ── drawing primitives ────────────────────────────────────────────

    def _panel(self, rect, title=None, accent=BORDER):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, accent, rect, width=1, border_radius=10)
        inner = rect.inflate(-28, -28)
        if title:
            self.screen.blit(self.f_h2.render(title, True, TEXT),
                             (rect.x + 16, rect.y + 12))
            inner = pygame.Rect(rect.x + 16, rect.y + 44,
                                rect.width - 32, rect.height - 56)
        return inner

    def _text(self, s, pos, fnt=None, col=TEXT, anchor="tl"):
        fnt = fnt or self.f_txt
        img = fnt.render(s, True, col)
        r = img.get_rect()
        setattr(r, {"tl": "topleft", "tr": "topright", "c": "center",
                    "ml": "midleft", "mc": "midtop"}[anchor], pos)
        if anchor == "tr":
            r.topright = pos
        elif anchor == "c":
            r.center = pos
        elif anchor == "ml":
            r.midleft = pos
        else:
            r.topleft = pos
        self.screen.blit(img, r)
        return r

    def _button(self, name, rect, label, active=False, col=CYAN):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        fill = col if active else (PANEL_HI if hover else PANEL)
        txtcol = BG if active else (col if hover else TEXT)
        pygame.draw.rect(self.screen, fill, rect, border_radius=7)
        pygame.draw.rect(self.screen, col, rect, width=1, border_radius=7)
        self._text(label, rect.center, self.f_lbl, txtcol, "c")
        self._buttons[name] = rect

    # ── heatmap ───────────────────────────────────────────────────────

    def _heatmap_surface(self, lut):
        norm = np.clip((self.rd[self.frame] - self.db_min)
                       / (self.db_max - self.db_min), 0.0, 1.0)
        idx = (norm * 255).astype(np.uint8)
        rgb = lut[idx]                       # (range, dop, 3)
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def _draw_heatmap(self, rect, lut, accent, show_overlays):
        # Cache the scaled heatmap; rebuild only when frame / scale / size change.
        key = (self.frame, self.db_min, self.db_max, rect.size)
        if getattr(self, "_hm_key", None) != key:
            self._hm_surf = pygame.transform.scale(
                self._heatmap_surface(lut), rect.size)
            self._hm_key = key
        self.screen.blit(self._hm_surf, rect.topleft)
        pygame.draw.rect(self.screen, accent, rect, width=1)

        def to_px(r_bin, d_shift):
            x = rect.x + (d_shift / self.n_dop) * rect.width
            y = rect.y + (r_bin / self.n_range) * rect.height
            return int(x), int(y)

        # Doppler / range axis ticks
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = rect.x + frac * rect.width
            hz = (frac - 0.5) * self.prf
            pygame.draw.line(self.screen, GRID, (x, rect.y), (x, rect.bottom), 1)
            self._text(f"{hz:+.0f}", (x, rect.bottom + 4), self.f_sm, MUTED, "mc")
            y = rect.y + frac * rect.height
            km = frac * self.n_range * self.range_res / 1000.0
            self._text(f"{km:.1f}", (rect.x - 6, y), self.f_sm, MUTED, "tr")
        self._text("Doppler (Hz) →", (rect.right, rect.bottom + 4),
                   self.f_sm, FAINT, "tr")
        self._text("Range (km) ↓", (rect.x - 6, rect.y - 2), self.f_sm, FAINT, "tr")

        if show_overlays and self.show_detections:
            for d in self.dets:
                px, py = to_px(d["r_bin"], d["d_bin"])
                pygame.draw.circle(self.screen, AMBER, (px, py), 3, 1)

        if show_overlays and self.show_tracks:
            cos_half = math.cos(math.radians(self.beta / 2.0))
            for tr in self.snapshot:
                if tr["state"] != "CONFIRMED":
                    continue
                r_m = tr["x"][0, 0]
                vr = tr["x"][1, 0]
                fd = (2.0 * vr * cos_half / self.lam) % self.prf
                if fd >= self.prf / 2:
                    fd -= self.prf
                d_bin = int(round(fd / (self.prf / self.cpi)))
                d_shift = (d_bin + self.n_dop // 2) % self.n_dop
                r_bin = r_m / self.range_res
                px, py = to_px(r_bin, d_shift)
                pygame.draw.rect(self.screen, CYAN, (px - 5, py - 5, 10, 10), 1)

        # Ground-truth target marker (always shown — the real missile cell)
        if self.truth_rd is not None:
            px, py = to_px(*self.truth_rd)
            col = GREEN if self.truth_detected else RED
            pygame.draw.line(self.screen, col, (px - 11, py), (px - 4, py), 2)
            pygame.draw.line(self.screen, col, (px + 4, py), (px + 11, py), 2)
            pygame.draw.line(self.screen, col, (px, py - 11), (px, py - 4), 2)
            pygame.draw.line(self.screen, col, (px, py + 4), (px, py + 11), 2)
            pygame.draw.circle(self.screen, col, (px, py), 13, 1)

    def _colorbar(self, rect, lut):
        for i in range(rect.height):
            v = 1.0 - i / rect.height
            c = lut[int(v * 255)]
            pygame.draw.line(self.screen, c, (rect.x, rect.y + i),
                             (rect.right, rect.y + i))
        pygame.draw.rect(self.screen, BORDER, rect, 1)
        self._text(f"{self.db_max:.0f}", (rect.right + 5, rect.y), self.f_sm, MUTED)
        self._text(f"{self.db_min:.0f} dB", (rect.right + 5, rect.bottom - 12),
                   self.f_sm, MUTED)

    # ── picket-fence geometry (shared by both modes) ──────────────────

    def _draw_picket(self, rect, accent, big=False):
        m = self.meta[self.frame]
        R = float(m["range_m"])
        alt = float(m["target_alt_m"])
        max_rng = self.n_range * self.range_res
        max_alt = max(80.0, self.drone_alt * 1.4)

        plot = rect.inflate(-24, -40)
        plot.y += 8
        sea_y = plot.bottom - 14

        def gx(downrange):
            return plot.x + (downrange / max_rng) * plot.width
        def gy(a):
            return sea_y - (a / max_alt) * (plot.height - 24)

        # sea
        pygame.draw.line(self.screen, (30, 70, 95), (plot.x, sea_y),
                         (plot.right, sea_y), 2)
        for sx in range(int(plot.x), int(plot.right), 26):
            pygame.draw.arc(self.screen, (24, 54, 74),
                            (sx, sea_y - 3, 26, 8), math.pi, 2 * math.pi, 1)
        self._text("SEA SURFACE", (plot.x + 2, sea_y + 4), self.f_sm, FAINT)

        # range gridlines
        for frac in (0.25, 0.5, 0.75, 1.0):
            x = gx(frac * max_rng)
            pygame.draw.line(self.screen, GRID, (x, plot.y), (x, sea_y), 1)
            self._text(f"{frac*max_rng/1000:.1f} km", (x, sea_y + 16),
                       self.f_sm, MUTED, "mc")

        # picket fence: active node + standby mesh nodes (honest labelling)
        px = gx(0)
        py = gy(self.drone_alt)
        for frac in (0.4, 0.72):
            nx = gx(frac * max_rng)
            ny = gy(self.drone_alt)
            pygame.draw.line(self.screen, FAINT, (nx, ny), (nx, sea_y), 1)
            pygame.draw.circle(self.screen, FAINT, (int(nx), int(ny)), 5, 1)
            self._text("MESH (standby)", (nx, ny - 15), self.f_sm, FAINT, "mc")

        # missile marker
        horiz = math.sqrt(max(R**2 - (self.drone_alt - alt)**2, 0.0))
        mx, my = gx(horiz), gy(alt)
        detected = self.truth_detected and self.truth_in

        # line of sight
        pygame.draw.line(self.screen, accent if detected else FAINT,
                         (px, py), (mx, my), 1)

        # active picket drone
        pygame.draw.line(self.screen, accent, (px, py), (px, sea_y), 2)
        pygame.draw.circle(self.screen, accent, (int(px), int(py)), 7)
        pygame.draw.circle(self.screen, BG, (int(px), int(py)), 3)
        self._text("PICKET DRONE", (px, py - 18), self.f_sm, accent, "mc")
        self._text(f"{self.drone_alt:.0f} m", (px, py - 32), self.f_sm, MUTED, "mc")

        # missile + detection marker
        mcol = GREEN if detected else (RED if self.truth_in else FAINT)
        pts = [(mx, my - 7), (mx + 7, my), (mx, my + 7), (mx - 7, my)]
        pygame.draw.polygon(self.screen, mcol, pts)
        if detected:
            t = (pygame.time.get_ticks() // 120) % 6
            pygame.draw.circle(self.screen, GREEN, (int(mx), int(my)), 12 + t, 1)
            self._text("◉ DETECTING", (mx, my + 14), self.f_sm, GREEN, "mc")
        elif self.truth_in:
            self._text("target (in clutter)", (mx, my + 14), self.f_sm, RED, "mc")
        self._text("MISSILE", (mx, my - 18), self.f_sm, mcol, "mc")

    # ── status helpers ────────────────────────────────────────────────

    def _status(self):
        if self.truth_detected:
            return "TARGET TRACKED", GREEN
        if self.truth_in:
            return "SEARCHING — TARGET IN CLUTTER", AMBER
        return "NO TARGET IN COVERAGE", MUTED

    def _confidence(self):
        if not self.truth_detected:
            return "—", MUTED
        if self.snr_eff_db >= 13:
            return "HIGH", GREEN
        if self.snr_eff_db >= 6:
            return "MODERATE", AMBER
        return "MARGINAL", AMBER

    def _threat(self):
        m = self.meta[self.frame]
        v = abs(float(m["velocity_mps"]))
        alt = float(m["target_alt_m"])
        if v > 500:                 # supersonic
            return "HIGH"
        if alt < 60:                # sea-skimming (low, hard to see)
            return "MEDIUM"
        return "LOW"

    # ── master draw ───────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(BG)
        self._buttons.clear()
        self._draw_header()
        if self.mode == "presentation":
            self._draw_presentation()
        else:
            self._draw_technical()
        self._draw_footer()

    def _draw_header(self):
        accent = CYAN if self.mode == "presentation" else GREEN
        self._text("SENTINEL MESH", (24, 18), self.f_h1, WHITE)
        self._text("AUTONOMOUS MARITIME RADAR PICKET", (26, 50), self.f_sm, MUTED)
        # mode toggle
        self._button("mode", pygame.Rect(self.W - 250, 22, 226, 34),
                     f"VIEW: {self.mode.upper()}", active=True, col=accent)
        # dataset name
        self._text(self.path.name, (self.W / 2, 26), self.f_sm, MUTED, "mc")
        self._text(f"frame {self.frame+1}/{self.n_frames}  ·  "
                   f"t = {self.meta[self.frame]['t_s']:.2f} s",
                   (self.W / 2, 44), self.f_sm, FAINT, "mc")

    def _draw_footer(self):
        bar = pygame.Rect(24, self.H - 30, self.W - 320, 12)
        pygame.draw.rect(self.screen, PANEL, bar, border_radius=6)
        prog = bar.width * (self.frame / max(self.n_frames - 1, 1))
        accent = CYAN if self.mode == "presentation" else GREEN
        pygame.draw.rect(self.screen, accent,
                         (bar.x, bar.y, prog, bar.height), border_radius=6)
        knob = (bar.x + prog, bar.centery)
        pygame.draw.circle(self.screen, WHITE, (int(knob[0]), int(knob[1])), 7)
        self._buttons["timeline"] = bar
        self._button("play", pygame.Rect(self.W - 280, self.H - 38, 90, 26),
                     "PAUSE" if self.playing else "PLAY", col=accent)
        self._button("prev", pygame.Rect(self.W - 180, self.H - 38, 36, 26), "‹", col=accent)
        self._button("next", pygame.Rect(self.W - 138, self.H - 38, 36, 26), "›", col=accent)
        self._text("TAB switch view · SPACE play · ←/→ step · [ ] dataset",
                   (24, self.H - 48), self.f_sm, FAINT)

    # ── PRESENTATION ──────────────────────────────────────────────────

    def _draw_presentation(self):
        m = self.meta[self.frame]
        top = 78
        # Status banner
        banner = pygame.Rect(24, top, self.W - 48, 64)
        label, col = self._status()
        pygame.draw.rect(self.screen, PANEL, banner, border_radius=10)
        pygame.draw.rect(self.screen, col, banner, width=2, border_radius=10)
        pygame.draw.circle(self.screen, col, (banner.x + 34, banner.centery), 12)
        self._text(label, (banner.x + 60, banner.centery), self.f_mid, col, "ml")
        conf, ccol = self._confidence()
        self._text("DETECTION CONFIDENCE", (banner.right - 220, banner.y + 14),
                   self.f_sm, MUTED)
        self._text(conf, (banner.right - 220, banner.y + 30), self.f_h2, ccol)

        body_y = top + 80
        # Left: plain-language target card
        card = pygame.Rect(24, body_y, 560, 360)
        inner = self._panel(card, "TARGET SUMMARY", CYAN)
        rows = [
            ("Range to picket", f"{m['range_m']/1000:.2f} km"),
            ("Closing speed", f"{abs(m['velocity_mps']):.0f} m/s  (Mach {abs(m['velocity_mps'])/340.29:.2f})"),
            ("Altitude", f"{m['target_alt_m']:.0f} m above sea  (sea-skimming)"),
            ("Heading", "inbound, 090° (true east)"),
            ("Radar signature", f"{m['rcs_dbsm']:.0f} dBsm  (small, missile-class)"),
            ("Threat assessment", self._threat()),
        ]
        y = inner.y + 6
        for lbl, val in rows:
            self._text(lbl, (inner.x, y), self.f_sm, MUTED)
            tcol = THREAT_COL.get(val, TEXT) if lbl == "Threat assessment" else TEXT
            self._text(val, (inner.x, y + 18), self.f_h2, tcol)
            y += 56

        # Right: picket-fence engagement profile
        prof = pygame.Rect(600, body_y, self.W - 624, 360)
        pin = self._panel(prof, "ENGAGEMENT PROFILE  ·  forward picket vs sea-skimmer", CYAN)
        self._draw_picket(pin, CYAN, big=True)

        # Bottom explainer strip
        strip = pygame.Rect(24, body_y + 376, self.W - 48, 86)
        sin = self._panel(strip, None, BORDER)
        explain = ("A forward picket drone watches the sea surface for incoming "
                   "missiles. The marker shows where the radar is currently "
                   "detecting the target. Green = confidently tracked; "
                   "red = present but buried in sea clutter.")
        # word-wrap
        words, line, lines = explain.split(), "", []
        for w in words:
            if self.f_txt.size(line + w)[0] > strip.width - 60:
                lines.append(line); line = ""
            line += w + " "
        lines.append(line)
        for i, ln in enumerate(lines):
            self._text(ln, (sin.x, sin.y + 4 + i * 22), self.f_txt, MUTED)

    # ── TECHNICAL ─────────────────────────────────────────────────────

    def _draw_technical(self):
        top = 78
        # Heatmap panel (left)
        hp = pygame.Rect(24, top, 620, self.H - top - 56)
        hin = self._panel(hp, "RANGE–DOPPLER MAP", GREEN)
        cbar = pygame.Rect(hin.right - 16, hin.y + 4, 12, hin.height - 52)
        hmap = pygame.Rect(hin.x, hin.y + 4, hin.width - 70, hin.height - 52)
        self._draw_heatmap(hmap, self.lut_green, GREEN, show_overlays=True)
        self._colorbar(cbar, self.lut_green)
        # legend (below the axis tick labels)
        lx = hin.x
        ly = hmap.bottom + 24
        self._legend_dot(lx, ly, RED, "truth (missed)")
        self._legend_dot(lx + 130, ly, GREEN, "truth (detected)")
        self._legend_dot(lx + 280, ly, AMBER, "CFAR hit")
        self._legend_dot(lx + 390, ly, CYAN, "track")

        # Telemetry (middle)
        tp = pygame.Rect(660, top, 470, 416)
        tin = self._panel(tp, "TELEMETRY  ·  ground truth", GREEN)
        m = self.meta[self.frame]
        det_txt, det_col = ("DETECTED", GREEN) if self.truth_detected else (
            ("IN CLUTTER", AMBER) if self.truth_in else ("OUT OF COVERAGE", MUTED))
        rows = [
            ("Detection status", det_txt, det_col),
            ("Slant range", f"{m['range_m']:.1f} m  (bin {m['range_bin']})", TEXT),
            ("Radial velocity", f"{m['velocity_mps']:+.1f} m/s", TEXT),
            ("True Doppler", f"{m['doppler_hz']:+.0f} Hz", TEXT),
            ("Aliased Doppler", f"{m['doppler_hz_aliased']:+.1f} Hz  ({m['doppler_wraps']} wraps)", TEXT),
            ("Single-pulse SNR", f"{m['snr_single_pulse_db']:.1f} dB", TEXT),
            (f"Integrated SNR (+{self.int_gain_db:.0f})", f"{self.snr_eff_db:.1f} dB", TEXT),
            ("RCS (Swerling)", f"{m['rcs_dbsm']:.2f} dBsm", TEXT),
            ("Multipath factor", f"{m['multipath_factor_db']:+.2f} dB", TEXT),
            ("Target altitude", f"{m['target_alt_m']:.1f} m", TEXT),
        ]
        y = tin.y + 2
        for lbl, val, c in rows:
            self._text(lbl, (tin.x, y), self.f_sm, MUTED)
            self._text(val, (tin.right, y), self.f_txt, c, "tr")
            y += 36

        # Detection / CFAR stats (middle-bottom)
        dp = pygame.Rect(660, top + 430, 470, self.H - top - 430 - 56)
        din = self._panel(dp, "DETECTOR", GREEN)
        cos_conf = sum(1 for t in self.snapshot if t["state"] == "CONFIRMED")
        stats = [
            ("CFAR algorithm", self.cfar_algo),
            ("Design Pfa", f"{self.pfa:.0e}"),
            ("CFAR hits this frame", str(self.n_dets)),
            ("Detection clusters", str(len(self.dets))),
            ("Confirmed tracks", str(cos_conf)),
            ("Bistatic angle β", f"{self.beta:.0f}°"),
            ("PRF / CPI", f"{self.prf:.0f} Hz / {self.cpi}"),
            ("Range resolution", f"{self.range_res:.1f} m"),
        ]
        y = din.y + 2
        for lbl, val in stats:
            self._text(lbl, (din.x, y), self.f_sm, MUTED)
            self._text(val, (din.right, y), self.f_txt, TEXT, "tr")
            y += 30

        # Picket geometry + controls (right)
        gp = pygame.Rect(1146, top, self.W - 1146 - 24, 300)
        gin = self._panel(gp, "PICKET GEOMETRY", GREEN)
        self._draw_picket(gin, GREEN)

        cp = pygame.Rect(1146, top + 314, self.W - 1146 - 24, self.H - top - 314 - 56)
        self._panel(cp, "CONTROLS & RUNS", GREEN)
        self._button("det", pygame.Rect(cp.x + 16, cp.y + 44, 120, 28),
                     f"DET {'ON' if self.show_detections else 'OFF'}",
                     active=self.show_detections, col=AMBER)
        self._buttons.pop("det", None)  # toggled via key D; show state only
        self._button("trk", pygame.Rect(cp.x + 146, cp.y + 44, 120, 28),
                     f"TRK {'ON' if self.show_tracks else 'OFF'}",
                     active=self.show_tracks, col=CYAN)
        self._buttons.pop("trk", None)
        self._text("D detections · T tracks · C cfar algo",
                   (cp.x + 16, cp.y + 80), self.f_sm, FAINT)
        self._draw_runlist(pygame.Rect(cp.x + 16, cp.y + 104,
                                       cp.width - 32, cp.height - 120))

    def _legend_dot(self, x, y, col, label):
        pygame.draw.circle(self.screen, col, (x + 5, y + 6), 4, 1)
        self._text(label, (x + 14, y), self.f_sm, MUTED)

    def _draw_runlist(self, rect, compact=False):
        if not compact:
            self._text("DATASETS  [ / ]", (rect.x, rect.y), self.f_sm, MUTED)
            rect = pygame.Rect(rect.x, rect.y + 18, rect.width, rect.height - 18)
        n_show = min(len(self.files), max(1, rect.height // 26)) if not compact else min(6, len(self.files))
        start = max(0, min(self.file_idx - n_show // 2, len(self.files) - n_show))
        for i in range(start, min(start + n_show, len(self.files))):
            ry = rect.y + (i - start) * 24
            r = pygame.Rect(rect.x, ry, rect.width, 22)
            active = i == self.file_idx
            name = self.files[i].stem.replace("cruise_run_", "RUN ")[:26]
            if active:
                pygame.draw.rect(self.screen, PANEL_HI, r, border_radius=5)
            self._text(("● " if active else "  ") + name, (r.x + 4, r.y + 3),
                       self.f_sm, (CYAN if active else MUTED))
            self._buttons[f"run:{i}"] = r


def main():
    p = argparse.ArgumentParser(description="SENTINEL Mesh radar console.")
    p.add_argument("--data_dir", type=str, default="integrated_output")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir.resolve()}")
        return 1
    Console(data_dir).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
SENTINEL Mesh: "Polished Presentation Mode" Cinematic Tactical Radar Console
=============================================================================
A premium, highly polished visualizer console. Redesigned to immediately
communicate autonomous cooperative drone tracking to a non-technical audience.
Features dynamic sector sweep lines, AWACS-style bracket HUD targets,
a top-down Swarm Triangulation overlap map, large target status readouts,
and custom CRT scanline overlays.

Author: Antigravity AI
Date: May 2026
"""

from __future__ import annotations

import os
import sys
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndimage
import pygame
from PIL import Image as PILImage

# ═══════════════════════════════════════════════════════════════════════════
# 1. Colors & Design Constants (Disciplined Professional Palette)
# ═══════════════════════════════════════════════════════════════════════════

# Debug Mode: Green-Phosphor Aesthetics
CLR_BG_DARK = (5, 9, 7)            # Deep dark engineering backdrop
CLR_BG_PANEL = (10, 20, 16)        # Engineering panel backdrop
CLR_PANEL_BORDER = (22, 40, 32)    # Muted border green
CLR_GRID = (6, 25, 15)             # Muted grid lines
CLR_TEXT_GLOW = (0, 255, 140)      # High-contrast neon green/cyan
CLR_TEXT_MUTED = (50, 95, 75)      # Muted tactical green

# Presentation Mode: Cinematic Matte Charcoal & Marine Cyan Aesthetics
CLR_BG_PRES = (8, 9, 12)           # Dark charcoal/matte black
CLR_BG_PRES_PANEL = (13, 17, 24)   # Deep matte panel backdrop
CLR_BORDER_PRES = (22, 30, 42)     # Muted blue-gray panel border
CLR_PRES_GRID = (16, 25, 36)       # Muted blue-gray gridlines
CLR_CYAN_GLOW = (0, 200, 255)      # Rich cinematic cyan text glow
CLR_TEXT_PRES_MUTED = (55, 80, 110) # Soft tactical navy-cyan
CLR_GREEN_CONFIRMED = (0, 230, 110) # Emerald-green confirmed tracks
CLR_THREAT_CRIMSON = (255, 30, 60) # Vibrant red active threat reticle
CLR_DETECTION_AMBER = (255, 180, 0) # Uncertain detection alarms
CLR_HUD_RED = (255, 75, 75)        # Warning readouts
CLR_TRACK_BOX = (0, 255, 255)      # Cyan active tracking overlay
CLR_COOP_RING = (230, 200, 50)     # Swarm ring gold

CLR_WHITE = (255, 255, 255)
CLR_BLACK = (0, 0, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Vectorized CFAR Layer
# ═══════════════════════════════════════════════════════════════════════════

class VectorizedCFAR:
    """Highly optimized vectorized CFAR detector using SciPy and NumPy."""
    @staticmethod
    def detect_ca_cfar(
        power_linear: np.ndarray,
        guard: int = 2,
        training: int = 8,
        pfa: float = 1e-4
    ) -> np.ndarray:
        """Vectorized Cell-Averaging CFAR along Doppler axis (axis 1) with wrapping."""
        win_total = 2 * (guard + training) + 1
        win_guard = 2 * guard + 1
        
        sum_total = ndimage.uniform_filter1d(power_linear, size=win_total, axis=1, mode='wrap') * win_total
        sum_guard = ndimage.uniform_filter1d(power_linear, size=win_guard, axis=1, mode='wrap') * win_guard
        
        sum_train = sum_total - sum_guard
        N = 2 * training
        noise_est = sum_train / N
        
        alpha = N * (pfa ** (-1.0 / N) - 1.0)
        threshold = noise_est * alpha
        
        return (power_linear > threshold).astype(np.int32)

    @staticmethod
    def detect_os_cfar(
        power_linear: np.ndarray,
        guard: int = 2,
        training: int = 8,
        pfa: float = 1e-4,
        k_ratio: float = 0.75
    ) -> np.ndarray:
        """Vectorized Ordered-Statistic CFAR using sliding window views and partial sort."""
        N_range, N_doppler = power_linear.shape
        half_w = guard + training
        
        padded = np.pad(power_linear, ((0, 0), (half_w, half_w)), mode='wrap')
        
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (1, 2 * half_w + 1), axis=(0, 1)).squeeze(axis=2)
        
        left_train = windows[:, :, :training]
        right_train = windows[:, :, training + 2 * guard + 1:]
        train_cells = np.concatenate([left_train, right_train], axis=2)
        
        N = 2 * training
        k = int(round(k_ratio * N))
        k = min(max(k, 1), N)
        
        os_vals = np.partition(train_cells, k - 1, axis=2)[:, :, k - 1]
        
        alpha = (pfa ** (-1.0 / (N - k + 1))) - 1.0
        threshold = os_vals * alpha
        
        return (power_linear > threshold).astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Dynamic Quadrant-Shifting Smart Label Solver
# ═══════════════════════════════════════════════════════════════════════════

class SmartLabelSolver:
    """Computes collision-free label placements in screen space using bounding rect overlap checking.
    Priority gates labels during high coordinate density.
    """
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.drawn_rects: list[pygame.Rect] = []

    def clear(self):
        """Clears registered label footprints for the current frame."""
        self.drawn_rects.clear()

    def solve_label(
        self,
        screen: pygame.Surface,
        text_lines: list[str],
        target_x: int,
        target_y: int,
        color: tuple[int, int, int],
        bg_color: tuple[int, int, int] | None = None,
        quadrants: list[str] | None = None
    ) -> bool:
        """Finds a collision-free quadrant for text lines, draws it, and saves its rect.
        Returns True if drawn successfully, or False if hidden due to crowding.
        """
        if quadrants is None:
            quadrants = ["RT", "LT", "RB", "LB"]
            
        rendered_lines = [self.font.render(line, True, color) for line in text_lines]
        line_height = self.font.get_height()
        w = max([img.get_width() for img in rendered_lines])
        h = len(rendered_lines) * line_height
        
        # Add padding around bounding rect
        w += 8
        h += 6
        
        best_rect = None
        best_quadrant = None
        
        ox = 14
        oy = 14
        
        for quad in quadrants:
            if quad == "RT":
                rx = target_x + ox
                ry = target_y - h - oy
            elif quad == "LT":
                rx = target_x - w - ox
                ry = target_y - h - oy
            elif quad == "RB":
                rx = target_x + ox
                ry = target_y + oy
            else:  # "LB"
                rx = target_x - w - ox
                ry = target_y + oy
                
            candidate_rect = pygame.Rect(rx, ry, w, h)
            
            # Enforce bounding area limits: must stay inside the Range-Doppler viewport bounds
            # Viewport bounds: (40, 95, 512, 640)
            if not (40 <= candidate_rect.left and candidate_rect.right <= 552 and
                    95 <= candidate_rect.top and candidate_rect.bottom <= 735):
                continue
            
            # Check overlap with existing label bounds
            collision = False
            for rect in self.drawn_rects:
                if candidate_rect.colliderect(rect):
                    collision = True
                    break
                    
            if not collision:
                best_rect = candidate_rect
                best_quadrant = quad
                break
                
        # Fallback: if all quadrants collide, hide the label to avoid visual clutter
        if best_rect is None:
            return False
            
        # Draw background if requested
        if bg_color is not None:
            pygame.draw.rect(screen, bg_color, best_rect)
            pygame.draw.rect(screen, color, best_rect, 1)
            
        # Draw lines inside the box
        px = best_rect.x + 4
        py = best_rect.y + 3
        for img in rendered_lines:
            screen.blit(img, (px, py))
            py += line_height
            
        self.drawn_rects.append(best_rect)
        return True


# ═══════════════════════════════════════════════════════════════════════════
# 4. Multi-Target Kalman Tracker
# ═══════════════════════════════════════════════════════════════════════════

def extract_detections_centroids(
    cfar_map: np.ndarray,
    range_resolution_m: float,
    prf_hz: float,
    cpi_pulses: int,
    lambda_m: float,
    bistatic_angle_deg: float
) -> list[dict[str, Any]]:
    """Groups active CFAR pixels into target cluster centroids with Doppler wrapping."""
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
            "range_m": range_m,
            "vr_mps": vr_mps,
            "r_bin": mean_r_bin,
            "d_bin": mean_c_bin,
            "size": len(cluster)
        })
        
    return detections


class TargetTrack:
    """Represents a single persistent target track maintained by a Kalman Filter."""
    def __init__(self, track_id: int, init_r: float, init_vr: float):
        self.track_id = track_id
        
        self.x = np.array([[init_r], [init_vr]], dtype=float)
        
        self.P = np.array([
            [100.0, 0.0],
            [0.0, 25.0]
        ], dtype=float)
        
        self.state = "TENTATIVE"
        self.history: list[tuple[float, float]] = []
        self.hit_count = 1
        self.miss_count = 0
        self.age = 1

    def predict(self, dt: float, sigma_a: float = 3.0):
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=float)
        
        Q = np.array([
            [0.25 * (dt**4) * (sigma_a**2), 0.5 * (dt**3) * (sigma_a**2)],
            [0.5 * (dt**3) * (sigma_a**2), (dt**2) * (sigma_a**2)]
        ], dtype=float)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1

    def update(self, z_r: float, z_vr_unwrapped: float, r_noise: float = 3.0, vr_noise: float = 1.0):
        z = np.array([[z_r], [z_vr_unwrapped]], dtype=float)
        H = np.eye(2)
        R = np.diag([r_noise**2, vr_noise**2])
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        y = z - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        
        self.hit_count += 1
        self.miss_count = 0
        
        self.history.append((float(self.x[0, 0]), float(self.x[1, 0])))
        if len(self.history) > 60:
            self.history.pop(0)


class MultiTargetTracker:
    """Manages multi-target tracking, data association, and track lifecycle operations."""
    def __init__(self, prf_hz: float, lambda_m: float, bistatic_angle_deg: float):
        self.tracks: list[TargetTrack] = []
        self.next_track_id = 1
        self.prf_hz = prf_hz
        self.lambda_m = lambda_m
        self.bistatic_angle_deg = bistatic_angle_deg
        
        cos_half = math.cos(math.radians(bistatic_angle_deg / 2.0))
        self.v_ambig = prf_hz * lambda_m / (2.0 * cos_half)

    def process_frame(self, detections: list[dict[str, Any]], dt: float):
        """Performs predictive gate association, Doppler unwrapping, and track updates."""
        for track in self.tracks:
            track.predict(dt)

        gate_r = 150.0
        gate_v = 40.0

        matched_detections = set()
        matched_tracks = set()

        for t_idx, track in enumerate(self.tracks):
            pred_r = track.x[0, 0]
            pred_vr = track.x[1, 0]

            best_det_idx = -1
            best_dist = float('inf')
            best_unwrapped_vr = 0.0

            for d_idx, det in enumerate(detections):
                if d_idx in matched_detections:
                    continue

                det_r = det["range_m"]
                det_vr_aliased = det["vr_mps"]

                n = int(round((pred_vr - det_vr_aliased) / self.v_ambig))
                det_vr_unwrapped = det_vr_aliased + n * self.v_ambig

                dist = math.sqrt(((det_r - pred_r)/gate_r)**2 + ((det_vr_unwrapped - pred_vr)/gate_v)**2)
                
                if dist < 1.0 and dist < best_dist:
                    best_dist = dist
                    best_det_idx = d_idx
                    best_unwrapped_vr = det_vr_unwrapped

            if best_det_idx != -1:
                matched_tracks.add(t_idx)
                matched_detections.add(best_det_idx)
                det = detections[best_det_idx]
                track.update(det["range_m"], best_unwrapped_vr)

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.miss_count += 1
                track.P *= 1.1
                track.history.append((float(track.x[0, 0]), float(track.x[1, 0])))
                if len(track.history) > 60:
                    track.history.pop(0)

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                init_r = det["range_m"]
                init_vr_aliased = det["vr_mps"]
                
                n = 258 if init_vr_aliased >= 0 else -258
                init_vr_unwrapped = init_vr_aliased + n * self.v_ambig
                
                new_track = TargetTrack(self.next_track_id, init_r, init_vr_unwrapped)
                self.next_track_id += 1
                self.tracks.append(new_track)

        active_tracks = []
        for track in self.tracks:
            if track.state == "TENTATIVE" and track.hit_count >= 3:
                track.state = "CONFIRMED"
                
            if track.state == "CONFIRMED" and track.miss_count < 6:
                active_tracks.append(track)
            elif track.state == "TENTATIVE" and track.miss_count < 2:
                active_tracks.append(track)
                
        self.tracks = active_tracks


# ═══════════════════════════════════════════════════════════════════════════
# 5. Swarm Fusion Manager (Top-down Swarm Triangulation Coordinates)
# ═══════════════════════════════════════════════════════════════════════════

class SwarmFusionManager:
    """Manages simulated cooperative multi-drone radar surveillance."""
    def __init__(self):
        # Coordinates (x, y) on our 2D tactical map:
        # Picket Alpha-1 at X = 70px offset, Y = 130px
        # Picket Alpha-2 near center X = 230px, Y = 60px
        # Picket Alpha-3 near right X = 360px, Y = 160px
        self.drone_pos = [
            {"name": "ALPHA-1", "x": 60, "y": 140, "radius": 90, "angle_start": -35, "angle_end": 45},
            {"name": "ALPHA-2", "x": 240, "y": 55, "radius": 130, "angle_start": 35, "angle_end": 125},
            {"name": "ALPHA-3", "x": 370, "y": 120, "radius": 110, "angle_start": 105, "angle_end": 195}
        ]
        # Physical coordinates for 3D physics triangulations (in meters)
        self.drone_1_pos = np.array([-19990.0, 0.0, 50.0])
        self.drone_2_pos = np.array([-16000.0, 0.0, 100.0])
        self.drone_3_pos = np.array([-12000.0, 0.0, 80.0])

    def get_cooperative_ranges(self, target_true_x: float, target_true_z: float) -> dict[str, float]:
        """Calculates exact physical ranges from each picket drone to target."""
        tgt = np.array([target_true_x, 0.0, target_true_z])
        r1 = float(np.linalg.norm(tgt - self.drone_1_pos))
        r2 = float(np.linalg.norm(tgt - self.drone_2_pos))
        r3 = float(np.linalg.norm(tgt - self.drone_3_pos))
        return {"Drone 1": r1, "Drone 2": r2, "Drone 3": r3}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dynamic Color Mappings
# ═══════════════════════════════════════════════════════════════════════════

def generate_radar_color_lut_debug() -> np.ndarray:
    """Generates a high-fidelity 256-level green-cyan phosphor LUT."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        R = int(clip_val((v ** 4.5) * 220))
        G = int(clip_val(v * 215 + (v ** 2) * 40 if v > 0.05 else v * 150))
        B = int(clip_val((v ** 2.2) * 165 + (v ** 6) * 90 if v > 0.1 else v * 80))
        lut[i] = [R, G, B]
    return lut

def generate_radar_color_lut_presentation() -> np.ndarray:
    """Generates a cinematic 256-level monochrome cyan-blue gradient LUT."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        # Dark navy -> rich teal -> glowing cyan-white
        R = int(clip_val((v ** 4) * 165))
        G = int(clip_val((v ** 1.8) * 195 + (v ** 4) * 55))
        B = int(clip_val(v * 235 + (v ** 0.5) * 20 if v > 0.0 else v * 30))
        lut[i] = [R, G, B]
    return lut

def clip_val(val: float) -> float:
    return max(0.0, min(255.0, val))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main Visualizer Engine
# ═══════════════════════════════════════════════════════════════════════════

class RadarConsoleApp:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dataset_files = sorted(list(data_dir.glob("*.npz")))
        if not self.dataset_files:
            raise FileNotFoundError(f"No .npz radar datasets found in {data_dir.resolve()}")
            
        self.current_file_idx = 0
        
        self.ui_mode = "presentation"
        
        # Mode Transition Animation variables
        self.transition_start_time = 0.0
        self.transition_duration = 0.4
        self.transition_from = "presentation"
        self.transition_to = "presentation"
        
        self.load_dataset(self.dataset_files[self.current_file_idx])
        
        # Initialize PyGame
        pygame.init()
        pygame.display.set_caption("SENTINEL Mesh: Radar Tracking Console")
        self.screen = pygame.display.set_mode((1280, 800))
        self.clock = pygame.time.Clock()
        
        # Configure fonts
        self.font_title = pygame.font.Font(None, 28)
        self.font_hud = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        
        self.label_solver = SmartLabelSolver(self.font_small)
        
        # Animation & Playback States
        self.current_frame = 0
        self.is_playing = True
        self.playback_speed = 1.0
        self.last_frame_update_time = time.time()
        
        # Rotating Vector sweep line floating index (Doppler axis scan)
        self.sweep_x = 0.0
        
        # Background phosphor noise particles
        self.bg_particles = [
            {
                'pos': (np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)),
                'alpha': np.random.randint(20, 110)
            } for _ in range(30)
        ]
        
        self.db_min = 10.0
        self.db_max = 50.0
        self.persistence_decay = 0.82
        self.pfa = 1e-4
        self.cfar_algorithm = "CA-CFAR"
        self.show_detections = False
        self.show_tracking = True
        self.show_swarm = True
        
        # Video recording buffer
        self.is_recording = False
        self.recorded_frames: list[np.ndarray] = []
        
        # PyGame Persistence Surface
        self.heatmap_rect = pygame.Rect(40, 95, 512, 640)
        self.persistence_surf = pygame.Surface((self.heatmap_rect.width, self.heatmap_rect.height))
        self.persistence_surf.fill(CLR_BG_DARK)
        
        # Generate Color LUTs
        self.color_lut_debug = generate_radar_color_lut_debug()
        self.color_lut_pres = generate_radar_color_lut_presentation()

        self.swarm_mgr = SwarmFusionManager()

    def toggle_ui_mode(self):
        """Instantly toggles UI Mode and triggers scanning sweep transition."""
        self.transition_start_time = time.time()
        self.transition_from = self.ui_mode
        self.ui_mode = "debug" if self.ui_mode == "presentation" else "presentation"
        self.transition_to = self.ui_mode
        
        self.persistence_surf.fill(CLR_BG_DARK)
        print(f"UI Mode Toggled: -> {self.ui_mode.upper()}")

    def load_dataset(self, filepath: Path):
        """Loads and parses NPZ radar dataset."""
        print(f"Loading radar dataset: {filepath.name}")
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        
        self.rd_video = data["rd_video"]
        self.metadata = data["metadata"].tolist()
        self.radar_config = data["radar_config"].tolist()
        
        self.n_frames, self.n_ranges, self.n_dopplers = self.rd_video.shape
        self.current_frame = 0
        
        # Initialize Tracker
        self.tracker = MultiTargetTracker(
            prf_hz=float(self.radar_config["prf_hz"]),
            lambda_m=float(self.radar_config["lambda_m"]),
            bistatic_angle_deg=float(self.radar_config["bistatic_angle_deg"])
        )
        
        self.precompute_tracker_states()

    def precompute_tracker_states(self):
        """Precomputes tracker snapshots frame-by-frame for fluid timeline scrubbing."""
        self.frame_tracker_snapshots = []
        
        temp_tracker = MultiTargetTracker(
            prf_hz=float(self.radar_config["prf_hz"]),
            lambda_m=float(self.radar_config["lambda_m"]),
            bistatic_angle_deg=float(self.radar_config["bistatic_angle_deg"])
        )
        
        dt = float(self.metadata[1]["t_s"] - self.metadata[0]["t_s"]) if self.n_frames > 1 else 0.128
        
        for f in range(self.n_frames):
            power_db = self.rd_video[f]
            power_lin = 10.0 ** (power_db / 10.0)
            
            det_map = VectorizedCFAR.detect_ca_cfar(power_lin, guard=2, training=8, pfa=1e-4)
            dets = extract_detections_centroids(
                cfar_map=det_map,
                range_resolution_m=self.radar_config["range_resolution_m"],
                prf_hz=self.radar_config["prf_hz"],
                cpi_pulses=self.radar_config["cpi_pulses"],
                lambda_m=self.radar_config["lambda_m"],
                bistatic_angle_deg=self.radar_config["bistatic_angle_deg"]
            )
            
            temp_tracker.process_frame(dets, dt)
            
            snapshot = []
            for track in temp_tracker.tracks:
                snapshot.append({
                    "id": track.track_id,
                    "x": track.x.copy(),
                    "P": track.P.copy(),
                    "state": track.state,
                    "miss_count": track.miss_count,
                    "history": list(track.history)
                })
            self.frame_tracker_snapshots.append(snapshot)

    def run(self):
        """Application visual loop execution."""
        running = True
        while running:
            running = self.handle_events()
            self.update_playback()
            self.draw_dashboard()
            
            if self.is_recording:
                frame_data = pygame.surfarray.array3d(self.screen)
                self.recorded_frames.append(frame_data.swapaxes(0, 1))
                if len(self.recorded_frames) >= 300:
                    self.stop_recording()
            
            # Show Recording Indicator
            if self.is_recording:
                rec_circle_visible = (int(time.time() * 2) % 2 == 0)
                if rec_circle_visible:
                    pygame.draw.circle(self.screen, CLR_HUD_RED, (1240, 20), 8)
                txt_rec = self.font_hud.render("REC", True, CLR_HUD_RED)
                self.screen.blit(txt_rec, (1195, 13))

            self.draw_mode_transition()
            
            # Draw CRT Scanlines over entire screen
            self.draw_crt_scanlines()

            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()
        sys.exit(0)

    def handle_events(self) -> bool:
        """Processes keystrokes, timeline scrubbing, and button clicks."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_TAB:
                    self.toggle_ui_mode()
                elif event.key == pygame.K_SPACE:
                    self.is_playing = not self.is_playing
                elif event.key == pygame.K_LEFT:
                    self.current_frame = max(0, self.current_frame - 1)
                elif event.key == pygame.K_RIGHT:
                    self.current_frame = min(self.n_frames - 1, self.current_frame + 1)
                elif event.key == pygame.K_PAGEUP:
                    self.cycle_dataset(-1)
                elif event.key == pygame.K_PAGEDOWN:
                    self.cycle_dataset(1)
                elif event.key == pygame.K_r:
                    self.toggle_recording()
                elif event.key == pygame.K_t:
                    self.show_tracking = not self.show_tracking
                elif event.key == pygame.K_c:
                    self.show_detections = not self.show_detections
                elif event.key == pygame.K_s:
                    self.show_swarm = not self.show_swarm
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_click(event.pos)
                    
        return True

    def cycle_dataset(self, offset: int):
        """Cycles loaded NPZ datasets."""
        self.current_file_idx = (self.current_file_idx + offset) % len(self.dataset_files)
        self.load_dataset(self.dataset_files[self.current_file_idx])
        self.persistence_surf.fill(CLR_BG_DARK)

    def toggle_recording(self):
        """Starts/stops frame captures for GIF encoding."""
        if not self.is_recording:
            print("Recording started...")
            self.is_recording = True
            self.recorded_frames = []
        else:
            self.stop_recording()

    def stop_recording(self):
        """Compiles recorded frame cache into loops animated GIF."""
        self.is_recording = False
        if not self.recorded_frames:
            return
            
        print("Compiling GIF export... Please wait.")
        export_dir = Path("./exports")
        export_dir.mkdir(exist_ok=True)
        
        filename = export_dir / f"radar_console_capture_{int(time.time())}.gif"
        
        self.screen.fill(CLR_BG_DARK)
        txt = self.font_title.render("ENCODING ANIMATED GIF... PLEASE WAIT", True, CLR_TEXT_GLOW)
        self.screen.blit(txt, (450, 380))
        pygame.display.flip()
        
        pil_images = []
        for frame in self.recorded_frames:
            img = PILImage.fromarray(frame)
            w, h = img.size
            img_scaled = img.resize((int(w * 0.6), int(h * 0.6)), PILImage.Resampling.BILINEAR)
            pil_images.append(img_scaled)
            
        if pil_images:
            pil_images[0].save(
                filename,
                save_all=True,
                append_images=pil_images[1:],
                duration=80,
                loop=0
            )
            print(f"Radar Console GIF exported: {filename.resolve()}")
            
        self.recorded_frames = []

    def update_playback(self):
        """Controls frame updates based on system clocks with a robust accumulator."""
        now = time.time()
        dt_elapsed = now - self.last_frame_update_time
        self.last_frame_update_time = now
        
        # Smooth continuous radar sweep glide index increment (3.0 seconds period)
        self.sweep_x = (self.sweep_x + dt_elapsed * 0.33) % 1.0

        if not self.is_playing:
            return
            
        if not hasattr(self, "accumulated_time"):
            self.accumulated_time = 0.0
            
        self.accumulated_time += dt_elapsed
        
        cpi_duration = float(self.metadata[1]["t_s"] - self.metadata[0]["t_s"]) if self.n_frames > 1 else 0.128
        frame_duration = cpi_duration / self.playback_speed
        
        if self.accumulated_time >= frame_duration:
            frames_to_advance = int(self.accumulated_time / frame_duration)
            self.current_frame = (self.current_frame + frames_to_advance) % self.n_frames
            self.accumulated_time %= frame_duration

    def handle_click(self, pos: tuple[int, int]):
        """Handles screen button triggers and slider clicks."""
        x, y = pos
        
        # Timeline scrubber region: [X: 40 to 1240, Y: 755 to 770]
        if 40 <= x <= 1240 and 750 <= y <= 775:
            pct = (x - 40) / 1200.0
            self.current_frame = int(pct * (self.n_frames - 1))
            self.current_frame = min(max(self.current_frame, 0), self.n_frames - 1)
            
        # Button: Play / Pause [X: 580 to 660, Y: 705 to 735]
        elif 580 <= x <= 660 and 705 <= y <= 735:
            self.is_playing = not self.is_playing
            
        # Button: Speed - [X: 520 to 570, Y: 705 to 735]
        elif 520 <= x <= 570 and 705 <= y <= 735:
            self.playback_speed = max(0.25, self.playback_speed - 0.25)
            
        # Button: Speed + [X: 670 to 720, Y: 705 to 735]
        elif 670 <= x <= 720 and 705 <= y <= 735:
            self.playback_speed = min(4.0, self.playback_speed + 0.25)

        # Button: Toggle Tracking [X: 1100 to 1240, Y: 425 to 455]
        elif 1100 <= x <= 1240 and 425 <= y <= 455:
            self.show_tracking = not self.show_tracking

        # Button: Toggle Detections [X: 1100 to 1240, Y: 460 to 490]
        elif 1100 <= x <= 1240 and 460 <= y <= 490:
            if self.ui_mode == "debug":
                self.show_detections = not self.show_detections

        # Button: Toggle Swarm [X: 1100 to 1240, Y: 495 to 525]
        elif 1100 <= x <= 1240 and 495 <= y <= 525:
            self.show_swarm = not self.show_swarm

        # Button: Toggle CFAR Method [X: 1100 to 1240, Y: 530 to 560]
        elif 1100 <= x <= 1240 and 530 <= y <= 560:
            if self.ui_mode == "debug":
                self.cfar_algorithm = "OS-CFAR" if self.cfar_algorithm == "CA-CFAR" else "CA-CFAR"

        # Button: Export GIF [X: 1100 to 1240, Y: 565 to 595]
        elif 1100 <= x <= 1240 and 565 <= y <= 595:
            self.toggle_recording()

        # Dynamic Range sliders (Locked in Presentation mode)
        elif 1100 <= x <= 1240 and 635 <= y <= 655:
            if self.ui_mode == "debug":
                self.db_min = float(max(0.0, min(self.db_max - 5.0, (x - 1100) / 140.0 * 60.0)))
            
        elif 1100 <= x <= 1240 and 665 <= y <= 685:
            if self.ui_mode == "debug":
                self.db_max = float(max(self.db_min + 5.0, min(80.0, (x - 1100) / 140.0 * 80.0)))

        # Header Mode Switcher Click: Top Right [X: 840 to 980, Y: 10 to 40]
        elif 840 <= x <= 980 and 10 <= y <= 40:
            self.toggle_ui_mode()

    # ── Viewport clipping safeguards ──
    def safe_draw_area(self, surface: pygame.Surface, rect: pygame.Rect):
        """Sets a clipping mask on the canvas to enforce panel boundaries."""
        surface.set_clip(rect)

    def clear_safe_area(self, surface: pygame.Surface):
        """Resets the surface clipping mask."""
        surface.set_clip(None)

    def draw_dashboard(self):
        """Renders dashboard viewports based on the active UI Mode."""
        bg_clr = CLR_BG_DARK if self.ui_mode == "debug" else CLR_BG_PRES
        self.screen.fill(bg_clr)
        
        # Enforce panel boundaries during drawing
        # 1. Header Viewport
        self.safe_draw_area(self.screen, pygame.Rect(0, 0, 1280, 50))
        self.draw_header()
        self.clear_safe_area(self.screen)
        
        # 2. Left Heatmap Viewport (Hero Panel - 55% space)
        self.safe_draw_area(self.screen, pygame.Rect(25, 75, 542, 675))
        self.draw_heatmap_panel()
        self.clear_safe_area(self.screen)
        
        # 3. Right Sidebar Control Viewport
        self.safe_draw_area(self.screen, pygame.Rect(1090, 75, 165, 675))
        self.draw_control_sidebar()
        self.clear_safe_area(self.screen)
        
        # 4. Telemetry Hud Viewport
        self.safe_draw_area(self.screen, pygame.Rect(592, 75, 482, 335))
        self.draw_hud_telemetry()
        self.clear_safe_area(self.screen)
        
        # 5. Tactical Swarm Triangulation Viewport
        self.safe_draw_area(self.screen, pygame.Rect(592, 425, 482, 325))
        self.draw_tactical_view()
        self.clear_safe_area(self.screen)
        
        # 6. Timeline Scrubber Navigation
        self.safe_draw_area(self.screen, pygame.Rect(25, 735, 1230, 60))
        self.draw_navigation_timeline()
        self.clear_safe_area(self.screen)

    def draw_header(self):
        """Draws top title panel and global console details."""
        panel_bg = CLR_BG_PANEL if self.ui_mode == "debug" else CLR_BG_PRES_PANEL
        panel_border = CLR_PANEL_BORDER if self.ui_mode == "debug" else CLR_BORDER_PRES
        text_color = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        text_muted = CLR_TEXT_MUTED if self.ui_mode == "debug" else CLR_TEXT_PRES_MUTED
        
        pygame.draw.rect(self.screen, panel_bg, (0, 0, 1280, 50))
        pygame.draw.line(self.screen, panel_border, (0, 50), (1280, 50), 2)
        
        lbl_title = self.font_title.render("SENTINEL MESH: COOPERATIVE RADAR MONITOR", True, text_color)
        self.screen.blit(lbl_title, (40, 15))
        
        file_info = f"ACTIVE TARGET LOG: {self.filepath.name}"
        lbl_file = self.font_hud.render(file_info, True, text_muted)
        self.screen.blit(lbl_file, (500, 18))
        
        btn_mode_rect = pygame.Rect(840, 10, 140, 30)
        mode_btn_bg = (20, 45, 30) if self.ui_mode == "debug" else (15, 35, 60)
        mode_btn_border = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        
        pygame.draw.rect(self.screen, mode_btn_bg, btn_mode_rect)
        pygame.draw.rect(self.screen, mode_btn_border, btn_mode_rect, 1)
        
        btn_txt = "DEBUG MODE" if self.ui_mode == "debug" else "PRESENTATION"
        lbl_btn = self.font_small.render(btn_txt, True, mode_btn_border)
        self.screen.blit(lbl_btn, (btn_mode_rect.centerx - lbl_btn.get_width()//2, btn_mode_rect.centery - lbl_btn.get_height()//2))

        fps_val = int(self.clock.get_fps())
        clock_txt = f"CONSOLE: 60Hz | FPS: {fps_val:02d}"
        lbl_fps = self.font_small.render(clock_txt, True, text_muted)
        self.screen.blit(lbl_fps, (1040, 20))

    def draw_heatmap_panel(self):
        """Main heatmap visualizer panel. Normalizes dB ranges, maps colormaps,

        blends persistent phosphor overlay surfaces, and draws sweep lines.
        """
        panel_bg = CLR_BG_PANEL if self.ui_mode == "debug" else CLR_BG_PRES_PANEL
        panel_border = CLR_PANEL_BORDER if self.ui_mode == "debug" else CLR_BORDER_PRES
        
        # Outer panel border
        pygame.draw.rect(self.screen, panel_bg, (25, 75, 542, 675))
        pygame.draw.rect(self.screen, panel_border, (25, 75, 542, 675), 2)
        
        # Extract dB power matrix for the active frame
        power_db = self.rd_video[self.current_frame]
        
        # Normalize and map values to colormap bounds
        norm_factor = (power_db - self.db_min) / (self.db_max - self.db_min)
        norm_factor = np.clip(norm_factor, 0.0, 1.0)
        
        # Color mapping (vectorized LUT execution)
        lut = self.color_lut_debug if self.ui_mode == "debug" else self.color_lut_pres
        rgb_data = lut[(norm_factor * 255).astype(np.uint8)]
        
        # Transpose to screen orientation (Doppler on X, Range on Y with closest at bottom)
        rgb_data = rgb_data[::-1, :, :]
        raw_surface = pygame.surfarray.make_surface(rgb_data.swapaxes(0, 1))
        
        upscaled_surf = pygame.transform.scale(raw_surface, (self.persistence_surf.get_width(), self.persistence_surf.get_height()))
        
        # Phosphor decay accumulation
        decay_mask = pygame.Surface((self.persistence_surf.get_width(), self.persistence_surf.get_height()), pygame.SRCALPHA)
        decay_mask.fill((0, 0, 0, int((1.0 - self.persistence_decay) * 255)))
        self.persistence_surf.blit(decay_mask, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        self.persistence_surf.blit(upscaled_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Blit phosphor display onto the main screen
        self.screen.blit(self.persistence_surf, (self.heatmap_rect.x, self.heatmap_rect.y))
        
        # ── Phosphor background particle noise ──
        h_x = self.heatmap_rect.x
        h_y = self.heatmap_rect.y
        h_w = self.heatmap_rect.width
        h_h = self.heatmap_rect.height
        
        for p in self.bg_particles:
            p['alpha'] += np.random.choice([-2, 2])
            p['alpha'] = max(10, min(100, p['alpha']))
            px = h_x + int(p['pos'][0] * h_w)
            py = h_y + int(p['pos'][1] * h_h)
            
            p_color = (0, 255, 140, p['alpha']) if self.ui_mode == "debug" else (0, 210, 255, p['alpha'])
            # Draw particle on main screen with alpha blend
            pt_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            pt_surf.fill(p_color)
            self.screen.blit(pt_surf, (px, py))

        # ── Rotating Radar Sweep Line with Trailing Phosphor Decay ──
        sweep_screen_x = h_x + int(self.sweep_x * h_w)
        
        # Draw sweeping vector line
        sweep_color = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        pygame.draw.line(self.screen, sweep_color, (sweep_screen_x, h_y), (sweep_screen_x, h_y + h_h), 2)
        
        # Gradient phosphor scan trail
        trail_w = 150
        trail_surf = pygame.Surface((trail_w, h_h), pygame.SRCALPHA)
        for tx in range(trail_w):
            alpha_val = int((1.0 - (tx / trail_w)) * 105)
            # Match mode colormap values
            seg_clr = (0, 255, 140, alpha_val) if self.ui_mode == "debug" else (0, 210, 255, alpha_val)
            pygame.draw.line(trail_surf, seg_clr, (trail_w - 1 - tx, 0), (trail_w - 1 - tx, h_h), 1)
            
        # Draw trailing segment behind the sweep line
        self.screen.blit(trail_surf, (sweep_screen_x - trail_w, h_y))

        # Draw linear grids
        self.draw_heatmap_grid()

        # Run Real-Time CFAR Detection
        power_lin = 10.0 ** (power_db / 10.0)
        if self.cfar_algorithm == "CA-CFAR":
            det_map = VectorizedCFAR.detect_ca_cfar(power_lin, guard=2, training=8, pfa=self.pfa)
        else:
            det_map = VectorizedCFAR.detect_os_cfar(power_lin, guard=2, training=8, pfa=self.pfa, k_ratio=0.75)

        # Draw raw alarm detections (Hidden in Presentation mode)
        if self.show_detections and self.ui_mode == "debug":
            self.draw_cfar_detections(det_map)
            
        # Draw Target Tracker Overlay
        if self.show_tracking:
            self.draw_tracker_overlay()

    def draw_heatmap_grid(self):
        """Draws dynamic grid lines and coordinate readouts on the heatmap."""
        h_x = self.heatmap_rect.x
        h_y = self.heatmap_rect.y
        h_w = self.heatmap_rect.width
        h_h = self.heatmap_rect.height
        
        grid_color = CLR_GRID if self.ui_mode == "debug" else CLR_PRES_GRID
        text_muted = CLR_TEXT_MUTED if self.ui_mode == "debug" else CLR_TEXT_PRES_MUTED
        
        # Doppler Vertical Lines
        doppler_ticks = [-400, -200, 0, 200, 400]
        for tick in doppler_ticks:
            pct = (tick + 500) / 1000.0
            x_pos = h_x + int(pct * h_w)
            pygame.draw.line(self.screen, grid_color, (x_pos, h_y), (x_pos, h_y + h_h), 1)
            
            lbl_d = self.font_small.render(f"{tick}Hz", True, text_muted)
            self.screen.blit(lbl_d, (x_pos - 15, h_y + h_h + 5))
            
        # Range Horizontal Lines
        range_ticks_m = [1000, 2000, 3000, 4000]
        for r_tick in range_ticks_m:
            pct = r_tick / 4500.0
            y_pos = h_y + h_h - int(pct * h_h)
            pygame.draw.line(self.screen, grid_color, (h_x, y_pos), (h_x + h_w, y_pos), 1)
            
            lbl_r = self.font_small.render(f"{r_tick / 1000.0:.1f}km", True, text_muted)
            self.screen.blit(lbl_r, (h_x - 38, y_pos - 6))

    def draw_cfar_detections(self, det_map: np.ndarray):
        """Highlights CFAR detection pixels with glowing amber boxes."""
        rows, cols = np.where(det_map > 0)
        h_x = self.heatmap_rect.x
        h_y = self.heatmap_rect.y
        h_w = self.heatmap_rect.width
        h_h = self.heatmap_rect.height
        
        max_draw = min(len(rows), 80)
        for i in range(max_draw):
            r = rows[i]
            d = cols[i]
            
            x_pct = d / float(self.n_dopplers)
            y_pct = r / float(self.n_ranges)
            
            x_pos = h_x + int(x_pct * h_w)
            y_pos = h_y + h_h - int(y_pct * h_h)
            
            pygame.draw.rect(self.screen, CLR_DETECTION_AMBER, (x_pos - 3, y_pos - 3, 6, 6), 1)

    def draw_tracker_overlay(self):
        """Draws target tracker models. Exposes different complexity, track limits,

        collision-free labels, and metrics in Presentation vs Debug mode.
        """
        snapshot = self.frame_tracker_snapshots[self.current_frame]
        
        h_x = self.heatmap_rect.x
        h_y = self.heatmap_rect.y
        h_w = self.heatmap_rect.width
        h_h = self.heatmap_rect.height
        
        cos_half = math.cos(math.radians(float(self.radar_config["bistatic_angle_deg"]) / 2.0))
        lambda_m = float(self.radar_config["lambda_m"])
        prf_hz = float(self.radar_config["prf_hz"])
        fd_res = prf_hz / self.n_dopplers
        
        self.label_solver.clear()
        
        # Track limits
        # Presentation: max 8 visible tracks. Debug: unlimited
        max_tracks = 8 if self.ui_mode == "presentation" else 999
        visible_count = 0
        
        # Filter tracks
        valid_tracks = []
        for track in snapshot:
            if self.ui_mode == "presentation" and track["state"] != "CONFIRMED":
                continue
            valid_tracks.append(track)
            
        # In Presentation mode, prioritize labels by threat level (descending)
        if self.ui_mode == "presentation":
            def get_threat_score(t):
                tr = t["x"][0, 0]
                tvr = t["x"][1, 0]
                return abs(tvr) / max(tr, 1.0)
            valid_tracks = sorted(valid_tracks, key=get_threat_score, reverse=True)

        for track in valid_tracks:
            if visible_count >= max_tracks:
                break
                
            r_m = track["x"][0, 0]
            vr_mps = track["x"][1, 0]
            
            r_bin = r_m / float(self.radar_config["range_resolution_m"])
            
            fd_true = 2.0 * vr_mps * cos_half / lambda_m
            fd_aliased = fd_true % prf_hz
            if fd_aliased > prf_hz / 2.0:
                fd_aliased -= prf_hz
            
            d_bin = (fd_aliased / fd_res) % self.n_dopplers
            
            x_pct = d_bin / float(self.n_dopplers)
            y_pct = r_bin / float(self.n_ranges)
            
            x_pos = h_x + int(x_pct * h_w)
            y_pos = h_y + h_h - int(y_pct * h_h)
            
            # ── Draw Target Markers ──
            # Presentation: cinematic HUD brackets. Debug: raw box.
            if self.ui_mode == "presentation":
                marker_color = CLR_THREAT_CRIMSON
                # Glowing reticle dot
                pygame.draw.circle(self.screen, marker_color, (x_pos, y_pos), 3)
                
                # AWACS fighter HUD corner brackets [ ]
                bracket_sz = 14
                # Top-Left
                pygame.draw.line(self.screen, marker_color, (x_pos - bracket_sz//2, y_pos - bracket_sz//2), (x_pos - bracket_sz//2 + 4, y_pos - bracket_sz//2), 2)
                pygame.draw.line(self.screen, marker_color, (x_pos - bracket_sz//2, y_pos - bracket_sz//2), (x_pos - bracket_sz//2, y_pos - bracket_sz//2 + 4), 2)
                # Top-Right
                pygame.draw.line(self.screen, marker_color, (x_pos + bracket_sz//2, y_pos - bracket_sz//2), (x_pos + bracket_sz//2 - 4, y_pos - bracket_sz//2), 2)
                pygame.draw.line(self.screen, marker_color, (x_pos + bracket_sz//2, y_pos - bracket_sz//2), (x_pos + bracket_sz//2, y_pos - bracket_sz//2 + 4), 2)
                # Bottom-Left
                pygame.draw.line(self.screen, marker_color, (x_pos - bracket_sz//2, y_pos + bracket_sz//2), (x_pos - bracket_sz//2 + 4, y_pos + bracket_sz//2), 2)
                pygame.draw.line(self.screen, marker_color, (x_pos - bracket_sz//2, y_pos + bracket_sz//2), (x_pos - bracket_sz//2, y_pos + bracket_sz//2 - 4), 2)
                # Bottom-Right
                pygame.draw.line(self.screen, marker_color, (x_pos + bracket_sz//2, y_pos + bracket_sz//2), (x_pos + bracket_sz//2 - 4, y_pos + bracket_sz//2), 2)
                pygame.draw.line(self.screen, marker_color, (x_pos + bracket_sz//2, y_pos + bracket_sz//2), (x_pos + bracket_sz//2, y_pos + bracket_sz//2 - 4), 2)
            else:
                marker_color = CLR_TRACK_BOX
                box_sz = 14
                pygame.draw.rect(self.screen, marker_color, (x_pos - box_sz//2, y_pos - box_sz//2, box_sz, box_sz), 2)
                
            # ── Smart Label Formatting ──
            label_lines = []
            
            if self.ui_mode == "presentation":
                phonetic = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL"]
                label_lines.append(f"TARGET {phonetic[(track['id'] - 1) % len(phonetic)]}")
                
                # Threat level dynamic checks
                meta = self.metadata[self.current_frame]
                alt_m = meta["target_alt_m"]
                
                if abs(vr_mps) > 600.0 and alt_m < 15.0:
                    threat_str = "HIGH THREAT"
                elif abs(vr_mps) > 300.0:
                    threat_str = "MED THREAT"
                else:
                    threat_str = "LOW THREAT"
                    
                # Dynamic Tracking Confidence from Kalman filter covariance
                trace_cov = float(track["P"][0, 0] + track["P"][1, 1])
                conf = max(10, min(99, int(100.0 - math.sqrt(trace_cov) * 4.0)))
                
                label_lines.append(f"THREAT: {threat_str}")
                label_lines.append(f"CONFIDENCE: {conf}%")
            else:
                label_lines.append(f"TRK #{track['id']} (CONFIRMED)")
                label_lines.append(f"R_COV : {track['P'][0, 0]:.1f}")
                label_lines.append(f"V_COV : {track['P'][1, 1]:.1f}")
                label_lines.append(f"MISSES: {track['miss_count']}")

            # Run Smart Label Solver
            bg_box_color = (13, 17, 24, 220) if self.ui_mode == "presentation" else (10, 20, 16, 220)
            self.label_solver.solve_label(self.screen, label_lines, x_pos, y_pos, marker_color, bg_box_color)

            # ── Draw Fading History Trails ──
            hist = track["history"]
            if len(hist) > 1:
                points = []
                for past_r, past_vr in hist:
                    past_r_bin = past_r / float(self.radar_config["range_resolution_m"])
                    past_fd_true = 2.0 * past_vr * cos_half / lambda_m
                    past_fd_aliased = past_fd_true % prf_hz
                    if past_fd_aliased > prf_hz / 2.0:
                        past_fd_aliased -= prf_hz
                    past_d_bin = (past_fd_aliased / fd_res) % self.n_dopplers
                    
                    p_x = h_x + int(past_d_bin / float(self.n_dopplers) * h_w)
                    p_y = h_y + h_h - int(past_r_bin / float(self.n_ranges) * h_h)
                    
                    if 0 <= p_x - h_x <= h_w and 0 <= p_y - h_y <= h_h:
                        points.append((p_x, p_y))
                
                if len(points) > 1:
                    for idx in range(len(points) - 1):
                        opacity_pct = (idx + 1) / len(points)
                        # Presentation: cinematic cyan/blue history. Debug: green history
                        if self.ui_mode == "presentation":
                            segment_clr = (int(0 * opacity_pct), int(210 * opacity_pct), int(255 * opacity_pct))
                        else:
                            segment_clr = (int(0 * opacity_pct), int(255 * opacity_pct), int(140 * opacity_pct))
                        pygame.draw.line(self.screen, segment_clr, points[idx], points[idx+1], 2)
                        
            visible_count += 1

    def draw_hud_telemetry(self):
        """Displays telemetry panels. Formats clean details in Presentation Mode

        vs complete engineering internals in Debug Mode.
        """
        panel_bg = CLR_BG_PANEL if self.ui_mode == "debug" else CLR_BG_PRES_PANEL
        panel_border = CLR_PANEL_BORDER if self.ui_mode == "debug" else CLR_BORDER_PRES
        text_color = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        text_muted = CLR_TEXT_MUTED if self.ui_mode == "debug" else CLR_TEXT_PRES_MUTED
        
        pygame.draw.rect(self.screen, panel_bg, (592, 75, 482, 335))
        pygame.draw.rect(self.screen, panel_border, (592, 75, 482, 335), 2)
        
        # Top-right title rename
        title_txt = "REAL-TIME TELEMETRY STATS" if self.ui_mode == "debug" else "ACTIVE TARGET STATUS"
        lbl_hud_title = self.font_title.render(title_txt, True, text_color)
        self.screen.blit(lbl_hud_title, (612, 90))
        
        meta = self.metadata[self.current_frame]
        
        lines = []
        if self.ui_mode == "presentation":
            # Simplified presentation statistics with spacing
            alt_m = meta["target_alt_m"]
            r_km = meta["range_m"] / 1000.0
            vr_mps = abs(meta["velocity_mps"])
            
            # Dynamic threat assignment
            if vr_mps > 600.0 and alt_m < 15.0:
                threat_txt = "CRITICAL (Supersonic Sea-Skimming Target)"
            elif vr_mps > 300.0:
                threat_txt = "HIGH (Supersonic Target)"
            else:
                threat_txt = "MEDIUM"
                
            # Dynamic confidence trace mapping
            snapshot = self.frame_tracker_snapshots[self.current_frame]
            if len(snapshot) > 0:
                cov_t = float(snapshot[0]["P"][0, 0] + snapshot[0]["P"][1, 1])
                conf = max(10, min(99, int(100.0 - math.sqrt(cov_t) * 4.0)))
            else:
                conf = 95
                
            # Output spaced readable telemetry layout
            lines = [
                f"Target Classification : TARGET ALPHA",
                f"Range                 : {r_km:.2f} km",
                f"Speed                 : {vr_mps:.1f} m/s (Mach {vr_mps / 340.29:.2f})",
                f"Altitude              : {alt_m:.1f} m (Sea-Skimming)",
                f"Heading               : 090° (True East)",
                f"Threat Level          : {threat_txt}",
                f"Tracking Confidence   : {conf}%",
                f"Cooperative Status    : CONFIRMED (Autonomous Swarm Lock)",
            ]
        else:
            lines = [
                f"SIMULATION TIME  : {meta['t_s']:.3f} s",
                f"PHYSICS TARGET X : {meta['range_m']*math.cos(math.radians(30)) - 19990.0:.1f} m",
                f"MISSILE ALTITUDE : {meta['target_alt_m']:.2f} m above MSL",
                f"RADAR SLANT RANGE: {meta['range_m']:.2f} m",
                f"CLOSING VELOCITY : {meta['velocity_mps']:.2f} m/s (Mach {meta['velocity_mps']/340.29:.2f})",
                f"SWERLING-1 RCS   : {meta['rcs_dbsm']:.2f} dBsm (Fluctuating)",
                f"MULTIPATH PROP F : {meta['multipath_factor_db']:.2f} dB (Fading Nulls)",
                f"TRUE RADAR DOPP  : {meta['doppler_hz']:.1f} Hz",
                f"ALIASED DOPPLER  : {meta['doppler_hz_aliased']:.2f} Hz",
                f"DOPPLER WRAP FRM : {meta['doppler_wraps']} full wraps",
                f"RADAR RESOLUTION : {self.radar_config['range_resolution_m']:.1f} m (Bin {meta['range_bin']})"
            ]
        
        y_pos = 125
        for line in lines:
            color = text_muted
            # High-contrast coloring
            if "CRITICAL" in line or "HIGH" in line:
                color = CLR_THREAT_CRIMSON
            elif "CONFIDENCE" in line:
                color = CLR_GREEN_CONFIRMED if self.ui_mode == "presentation" else CLR_TEXT_GLOW
            elif "CONFIRMED" in line:
                color = CLR_CYAN_GLOW
            elif "Altitude" in line and meta["target_alt_m"] < 10.0:
                color = CLR_WHITE
                
            lbl_line = self.font_hud.render(line, True, color)
            self.screen.blit(lbl_line, (612, y_pos))
            y_pos += 22

    def draw_tactical_view(self):
        """Displays top-down or side profile maps. Formats clean vector fans

        in Presentation Mode vs full picket grids in Debug Mode.
        """
        panel_bg = CLR_BG_PANEL if self.ui_mode == "debug" else CLR_BG_PRES_PANEL
        panel_border = CLR_PANEL_BORDER if self.ui_mode == "debug" else CLR_BORDER_PRES
        text_color = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        
        pygame.draw.rect(self.screen, panel_bg, (592, 425, 482, 325))
        pygame.draw.rect(self.screen, panel_border, (592, 425, 482, 325), 2)
        
        title_txt = "FORWARD PICKET TACTICAL PROFILE" if self.ui_mode == "debug" else "SWARM TACTICAL OVERVIEW MAP"
        lbl_tac_title = self.font_title.render(title_txt, True, text_color)
        self.screen.blit(lbl_tac_title, (612, 440))
        
        tac_rect = pygame.Rect(612, 475, 442, 220)
        pygame.draw.rect(self.screen, CLR_BG_DARK, tac_rect)
        pygame.draw.rect(self.screen, panel_border, tac_rect, 1)
        
        meta = self.metadata[self.current_frame]
        r_m = meta["range_m"]
        alt_m = meta["target_alt_m"]
        
        # ── Mode 1: Presentation Mode - Swarm Cooperative Intersection Map (Top-Down) ──
        if self.ui_mode == "presentation":
            # Swarm visual layout Registry
            swarm_pos = self.swarm_mgr.drone_pos
            
            # Draw radar coverage fan sectors (glowing translucent wedges)
            cone_surf = pygame.Surface((tac_rect.width, tac_rect.height), pygame.SRCALPHA)
            
            # Map scaling: 4.5km max range corresponds to visual width scale
            scale_sc = 150.0 / 4500.0
            target_true_x = r_m * math.cos(math.radians(30)) - 19990.0
            
            # Target visual top-down coordinates: centered relative to Drone ALPHA-1
            cx_target = swarm_pos[0]["x"] + int(target_true_x * scale_sc)
            cy_target = swarm_pos[0]["y"] - int(alt_m * scale_sc)
            
            # 1. Renders overlapping radar sweeps
            for drone in swarm_pos:
                dx, dy = drone["x"], drone["y"]
                dr_px = int(drone["radius"] * scale_sc * 1.5)
                
                # Alpha wedge scans
                # Angles in radians for sector projection
                ang_start = math.radians(drone["angle_start"])
                ang_end = math.radians(drone["angle_end"])
                
                # Draw translucent coverage wedge scans (green-cyan)
                pygame.draw.arc(cone_surf, (0, 210, 255, 15), (dx - dr_px, dy - dr_px, dr_px*2, dr_px*2), ang_start, ang_end, dr_px)
                pygame.draw.arc(cone_surf, (0, 210, 255, 30), (dx - dr_px, dy - dr_px, dr_px*2, dr_px*2), ang_start, ang_end, 2)
                
            # 2. Triangulate cooperative overlap zones
            if self.show_swarm:
                # Golden overlay showing swarm cooperative intersection
                # Centered where target sits between picket nodes
                over_r = 30
                pygame.draw.circle(cone_surf, (255, 200, 0, 20), (cx_target, cy_target), over_r)
                pygame.draw.circle(cone_surf, (255, 200, 0, 50), (cx_target, cy_target), over_r, 1)
                
                # Trilateration intersection lines
                for drone in swarm_pos:
                    dx, dy = drone["x"], drone["y"]
                    pygame.draw.line(cone_surf, (255, 200, 0, 45), (dx, dy), (cx_target, cy_target), 1)

            # Blit translucent overlays onto the panel surface
            self.screen.blit(cone_surf, (tac_rect.left, tac_rect.top))

            # 3. Draw static Picket Drones ALPHA-1, 2, 3
            for idx, drone in enumerate(swarm_pos):
                dx = tac_rect.left + drone["x"]
                dy = tac_rect.top + drone["y"]
                
                pygame.draw.circle(self.screen, CLR_GREEN_CONFIRMED, (dx, dy), 5)
                pygame.draw.circle(self.screen, CLR_WHITE, (dx, dy), 2)
                
                lbl_dname = self.font_small.render(drone["name"], True, CLR_GREEN_CONFIRMED)
                self.screen.blit(lbl_dname, (dx - 22, dy - 15))
                
            # 4. Pulsing target indicator
            tx = tac_rect.left + cx_target
            ty = tac_rect.top + cy_target
            
            if tac_rect.left <= tx <= tac_rect.right and tac_rect.top <= ty <= tac_rect.bottom:
                # Pulsing ring animation
                pulse_r = int((time.time() * 25) % 15) + 5
                pygame.draw.circle(self.screen, CLR_THREAT_CRIMSON, (tx, ty), pulse_r, 1)
                
                # Center threat dot
                pygame.draw.circle(self.screen, CLR_THREAT_CRIMSON, (tx, ty), 4)
                pygame.draw.circle(self.screen, CLR_WHITE, (tx, ty), 2)
                
                lbl_tgt_map = self.font_small.render("THREAT LOCK", True, CLR_THREAT_CRIMSON)
                self.screen.blit(lbl_tgt_map, (tx + 8, ty - 8))
                
        # ── Mode 2: Technical Debug Mode - Look-Down Geometry Grid ──
        else:
            # Draw Sea surface
            sea_y = tac_rect.bottom - 20
            pygame.draw.line(self.screen, (15, 60, 90), (tac_rect.left, sea_y), (tac_rect.right, sea_y), 2)
            for wx in range(tac_rect.left, tac_rect.right, 30):
                pygame.draw.arc(self.screen, (15, 60, 90), (wx, sea_y - 4, 15, 8), math.pi, 0, 1)
                
            drone_screen_x = tac_rect.left + 50
            drone_screen_y = sea_y - 120
            
            pygame.draw.circle(self.screen, CLR_TRACK_BOX, (drone_screen_x, drone_screen_y), 6)
            pygame.draw.line(self.screen, CLR_TRACK_BOX, (drone_screen_x - 10, drone_screen_y), (drone_screen_x + 10, drone_screen_y), 2)
            pygame.draw.line(self.screen, CLR_TRACK_BOX, (drone_screen_x, drone_screen_y), (drone_screen_x, drone_screen_y + 8), 2)
            
            lbl_dr = self.font_small.render("PICKET DRONE (50m ASL)", True, CLR_TRACK_BOX)
            self.screen.blit(lbl_dr, (drone_screen_x - 40, drone_screen_y - 20))
            
            # Target Missile
            dz = 50.0 - alt_m
            dx = math.sqrt(max(r_m**2 - dz**2, 1.0))
            scale_x = 350.0 / 4500.0
            
            target_screen_x = drone_screen_x + int(dx * scale_x)
            target_screen_y = sea_y - int(alt_m * (120.0 / 50.0))
            
            if tac_rect.left <= target_screen_x <= tac_rect.right:
                pygame.draw.polygon(self.screen, CLR_THREAT_CRIMSON, [
                    (target_screen_x, target_screen_y - 4),
                    (target_screen_x + 8, target_screen_y),
                    (target_screen_x, target_screen_y + 4),
                    (target_screen_x - 4, target_screen_y)
                ])
                pygame.draw.circle(self.screen, CLR_WHITE, (target_screen_x, target_screen_y), 2)
                
                lbl_tgt = self.font_small.render("MISSILE (TARGET)", True, CLR_THREAT_CRIMSON)
                self.screen.blit(lbl_tgt, (target_screen_x - 30, target_screen_y - 18))
                
                pygame.draw.line(self.screen, (20, 85, 55), (drone_screen_x, drone_screen_y), (target_screen_x, target_screen_y), 1)
                pygame.draw.line(self.screen, (10, 45, 60), (drone_screen_x, drone_screen_y), (target_screen_x, sea_y), 1)
                pygame.draw.line(self.screen, (10, 45, 60), (target_screen_x, sea_y), (target_screen_x, target_screen_y), 1)

            if self.show_swarm:
                drone_2_screen_x = drone_screen_x + int(3990 * scale_x)
                drone_2_screen_y = sea_y - 180
                
                pygame.draw.circle(self.screen, CLR_COOP_RING, (drone_2_screen_x, drone_2_screen_y), 5)
                lbl_dr2 = self.font_small.render("PICKET-2 (100m)", True, CLR_COOP_RING)
                self.screen.blit(lbl_dr2, (drone_2_screen_x - 35, drone_2_screen_y - 16))
                
                coop = self.swarm_mgr.get_cooperative_ranges(meta["range_m"]*math.cos(math.radians(30)) - 19990.0, meta["target_alt_m"])
                r1_sc = int(coop["Drone 1"] * scale_x)
                r2_sc = int(coop["Drone 2"] * scale_x)
                
                pygame.draw.circle(self.screen, (0, 100, 50), (drone_screen_x, drone_screen_y), r1_sc, 1)
                pygame.draw.circle(self.screen, (120, 100, 20), (drone_2_screen_x, drone_2_screen_y), r2_sc, 1)
                pygame.draw.circle(self.screen, CLR_WHITE, (target_screen_x, target_screen_y), 6, 1)

            # Grid lines
            for gx in range(tac_rect.left, tac_rect.right, 80):
                pygame.draw.line(self.screen, (10, 25, 20), (gx, tac_rect.top), (gx, tac_rect.bottom), 1)
            for gy in range(tac_rect.top, tac_rect.bottom, 40):
                pygame.draw.line(self.screen, (10, 25, 20), (tac_rect.left, gy), (tac_rect.right, gy), 1)

    def draw_control_sidebar(self):
        """Interactive control buttons panel on right side. Heavy simplification

        in Presentation Mode, hiding all low-level RF diagnostic components.
        """
        panel_bg = CLR_BG_PANEL if self.ui_mode == "debug" else CLR_BG_PRES_PANEL
        panel_border = CLR_PANEL_BORDER if self.ui_mode == "debug" else CLR_BORDER_PRES
        text_color = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        text_muted = CLR_TEXT_MUTED if self.ui_mode == "debug" else CLR_TEXT_PRES_MUTED
        
        pygame.draw.rect(self.screen, panel_bg, (1090, 75, 165, 675))
        pygame.draw.rect(self.screen, panel_border, (1090, 75, 165, 675), 2)
        
        lbl_ctrl = self.font_title.render("CONTROLS", True, text_color)
        self.screen.blit(lbl_ctrl, (1110, 90))
        
        # Simplified Interactive buttons in Presentation Mode
        buttons = []
        if self.ui_mode == "presentation":
            buttons = [
                ("MODE SWITCH", True, CLR_CYAN_GLOW, 425),
                ("TRACKING", self.show_tracking, CLR_GREEN_CONFIRMED, 460),
                ("SWARM VIEW", self.show_swarm, CLR_COOP_RING, 495),
                ("CAPTURE GIF" if not self.is_recording else "RECORDING...", self.is_recording, CLR_HUD_RED, 530)
            ]
        else:
            buttons = [
                ("MODE SWITCH", True, CLR_TEXT_GLOW, 425),
                ("TRACKING", self.show_tracking, CLR_TRACK_BOX, 460),
                ("DETECTIONS", self.show_detections, CLR_DETECTION_AMBER, 495),
                ("SWARM VIEW", self.show_swarm, CLR_COOP_RING, 530),
                (self.cfar_algorithm, True, text_color, 565),
                ("CAPTURE GIF" if not self.is_recording else "RECORDING...", self.is_recording, CLR_HUD_RED, 600)
            ]
        
        for name, active, base_color, y_pos in buttons:
            btn_rect = pygame.Rect(1100, y_pos, 145, 30)
            fill_color = CLR_BG_DARK if not active else (int(base_color[0]*0.15), int(base_color[1]*0.15), int(base_color[2]*0.15))
            border_color = base_color if active else panel_border
            
            pygame.draw.rect(self.screen, fill_color, btn_rect)
            pygame.draw.rect(self.screen, border_color, btn_rect, 1)
            
            lbl_btn = self.font_hud.render(name, True, border_color)
            txt_x = btn_rect.centerx - lbl_btn.get_width() // 2
            txt_y = btn_rect.centery - lbl_btn.get_height() // 2
            self.screen.blit(lbl_btn, (txt_x, txt_y))

        # Dynamic Range sliders - Excluded in Presentation Mode
        if self.ui_mode == "debug":
            lbl_sliders = self.font_hud.render("DB RANGE LIMITS", True, text_muted)
            self.screen.blit(lbl_sliders, (1100, 642))
            
            # Min Slider
            min_pct = self.db_min / 60.0
            pygame.draw.rect(self.screen, CLR_BG_DARK, (1100, 674, 140, 6))
            pygame.draw.rect(self.screen, panel_border, (1100, 674, 140, 6), 1)
            pygame.draw.circle(self.screen, text_color, (1100 + int(min_pct * 140), 677), 5)
            lbl_min = self.font_small.render(f"Min: {self.db_min:.0f}dB", True, text_muted)
            self.screen.blit(lbl_min, (1100, 660))

            # Max Slider
            max_pct = self.db_max / 80.0
            pygame.draw.rect(self.screen, CLR_BG_DARK, (1100, 704, 140, 6))
            pygame.draw.rect(self.screen, panel_border, (1100, 704, 140, 6), 1)
            pygame.draw.circle(self.screen, text_color, (1100 + int(max_pct * 140), 707), 5)
            lbl_max = self.font_small.render(f"Max: {self.db_max:.0f}dB", True, text_muted)
            self.screen.blit(lbl_max, (1100, 690))

        # Simulation dataset logs runs
        lbl_runs = self.font_hud.render("SIMULATION RUNS", True, text_muted)
        
        # Adjust vertical position to fit Presentation screen padding cleanly
        y_label = 125 if self.ui_mode == "debug" else 145
        self.screen.blit(lbl_runs, (1100, y_label))
        
        # Enclose dataset runs inside a beautiful border box
        runs_box_rect = pygame.Rect(1100, y_label + 20, 145, 190)
        pygame.draw.rect(self.screen, CLR_BG_DARK, runs_box_rect)
        pygame.draw.rect(self.screen, panel_border, runs_box_rect, 1)
        
        y_run = y_label + 26
        for idx, file in enumerate(self.dataset_files):
            if abs(idx - self.current_file_idx) > 5:
                continue
                
            color = text_color if idx == self.current_file_idx else text_muted
            prefix = "▶ " if idx == self.current_file_idx else "  "
            
            short_name = file.stem.replace("cruise_run_", "RUN ")[:14]
            lbl_r_name = self.font_small.render(f"{prefix}{short_name}", True, color)
            self.screen.blit(lbl_r_name, (1105, y_run))
            y_run += 16
            
        lbl_help = self.font_small.render("[PgUp/PgDn] Cycle Logs", True, text_muted)
        self.screen.blit(lbl_help, (1100, y_label + 220))

    def draw_navigation_timeline(self):
        """Horizontal dynamic navigation bar for scrubbing frame timeline."""
        panel_bg = CLR_BG_PANEL if self.ui_mode == "debug" else CLR_BG_PRES_PANEL
        panel_border = CLR_PANEL_BORDER if self.ui_mode == "debug" else CLR_BORDER_PRES
        text_color = CLR_TEXT_GLOW if self.ui_mode == "debug" else CLR_CYAN_GLOW
        text_muted = CLR_TEXT_MUTED if self.ui_mode == "debug" else CLR_TEXT_PRES_MUTED
        
        pygame.draw.rect(self.screen, panel_bg, (25, 755, 1230, 35))
        pygame.draw.rect(self.screen, panel_border, (25, 755, 1230, 35), 2)
        
        scrub_rect = pygame.Rect(40, 765, 1200, 15)
        pygame.draw.rect(self.screen, CLR_BG_DARK, scrub_rect)
        pygame.draw.rect(self.screen, panel_border, scrub_rect, 1)
        
        curr_pct = self.current_frame / (self.n_frames - 1)
        pygame.draw.rect(self.screen, text_color, (40, 765, int(curr_pct * 1200), 15))
        pygame.draw.circle(self.screen, CLR_WHITE, (40 + int(curr_pct * 1200), 772), 8)
        
        lbl_frm = self.font_small.render(f"FRAME: {self.current_frame:03d} / {self.n_frames - 1}", True, text_muted)
        self.screen.blit(lbl_frm, (50, 738))
        
        state_txt = "PLAYING" if self.is_playing else "PAUSED"
        state_clr = text_color if self.is_playing else CLR_HUD_RED
        lbl_state = self.font_hud.render(f"PLAYBACK: {state_txt} ({self.playback_speed:.2f}x)  |  [TAB] Switch Console Mode", True, state_clr)
        self.screen.blit(lbl_state, (500, 736))

    def draw_mode_transition(self):
        """Renders cinematic sweep scans / reboots during UI mode transitions."""
        t_elapsed = time.time() - self.transition_start_time
        if t_elapsed < self.transition_duration:
            pct = t_elapsed / self.transition_duration
            y_pos = int(pct * 800)
            
            sweep_surf = pygame.Surface((1280, 45), pygame.SRCALPHA)
            bar_color = CLR_TEXT_GLOW if self.transition_to == "debug" else CLR_CYAN_GLOW
            
            sweep_surf.fill((bar_color[0], bar_color[1], bar_color[2], int((1.0 - pct) * 150)))
            self.screen.blit(sweep_surf, (0, y_pos - 22))
            
            pygame.draw.line(self.screen, bar_color, (0, y_pos), (1280, y_pos), 3)
            
            alert_bg = (13, 17, 24) if self.transition_to == "presentation" else (10, 20, 16)
            alert_border = CLR_CYAN_GLOW if self.transition_to == "presentation" else CLR_TEXT_GLOW
            
            alert_rect = pygame.Rect(440, 360, 400, 80)
            pygame.draw.rect(self.screen, alert_bg, alert_rect)
            pygame.draw.rect(self.screen, alert_border, alert_rect, 2)
            
            cfg_lbl = "RECONFIGURING UI SYSTEM..." if self.transition_to == "presentation" else "REBOOTING DIAGNOSTIC ENGINE..."
            lbl_a1 = self.font_hud.render(cfg_lbl, True, alert_border)
            lbl_a2 = self.font_small.render("SWITCHING TERMINAL MODES • STANDBY", True, CLR_WHITE)
            
            self.screen.blit(lbl_a1, (alert_rect.centerx - lbl_a1.get_width()//2, alert_rect.y + 20))
            self.screen.blit(lbl_a2, (alert_rect.centerx - lbl_a2.get_width()//2, alert_rect.y + 48))

    def draw_crt_scanlines(self):
        """Draws dynamic visual scanlines over the final rendered screen for CRT feel."""
        # Horizontal lines every 4th pixel vertically with 5% alpha
        scanline_surf = pygame.Surface((1280, 800), pygame.SRCALPHA)
        for y in range(0, 800, 4):
            pygame.draw.line(scanline_surf, (0, 0, 0, 14), (0, y), (1280, y), 1)
        self.screen.blit(scanline_surf, (0, 0))


# ═══════════════════════════════════════════════════════════════════════════
# 8. Launcher Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "integrated_output"
    
    if not data_dir.exists():
        print(f"Error: Output folder '{data_dir.resolve()}' does not exist.")
        print("Please run trajectory integration to populate NPZ datasets first:")
        print("    python integrate_simulations.py")
        sys.exit(1)
        
    print("Launching SENTINEL Mesh Cinematic Radar Console...")
    print("Press SPACE to play/pause, TAB to switch Presentation / Debug modes instantly.")
    print("PAGE UP / PAGE DOWN cycles simulation runs.")
    
    app = RadarConsoleApp(data_dir)
    app.run()

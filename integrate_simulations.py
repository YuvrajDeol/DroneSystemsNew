"""
integrate_simulations.py — Trajectory-to-Radar Integration for SENTINEL Mesh
=============================================================================

Reads missile kinematic trajectory text files (produced by the physics
engine at 100 Hz), translates each trajectory into radar-domain
parameters (range, Doppler, RCS), generates frame-by-frame
Range-Doppler clutter maps with embedded targets, and serialises the
result as a 3D RD-video (Time × Range × Doppler) for downstream ML
training.

Geometry (Look-Down / Forward Picket)
-------------------------------------
The radar drone hovers at a forward picket position near the missile's
spawn point (default ``(-19990, 0, 50)`` m).  The missile launches at
``x ≈ -20000`` m and flies toward the mothership at ``x = 0``.  All
range and Doppler values are computed *relative to the drone*, not the
origin.  As the missile passes beneath the drone, radial velocity
crosses zero (Doppler notch) and becomes positive as it recedes.

Usage
-----
    python integrate_simulations.py                       # defaults
    python integrate_simulations.py --input_dir logs/     # custom path
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import trange

from clutter_model import (
    BistaticClutterModel,
    RadarSystemParams,
    embed_target_in_clutter,
    multipath_propagation_factor,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

C_LIGHT_MPS: float = 3.0e8            # speed of light  (m/s)
CARRIER_FREQ_HZ: float = 77.0e9       # SENTINEL Mesh carrier  (Hz)
LAMBDA_M: float = C_LIGHT_MPS / CARRIER_FREQ_HZ  # ≈ 3.896 mm

DEFAULT_BISTATIC_ANGLE_DEG: float = 90.0
DEFAULT_PRF_HZ: int = 1000
DEFAULT_CPI_PULSES: int = 128
DEFAULT_N_RANGE_BINS: int = 1500
DEFAULT_CNR_DB: float = 15.0
DEFAULT_RANGE_RES_M: float = 3.0      # range-bin resolution (m)  → 4.5 km max
DEFAULT_BASE_RCS_DBSM: float = -13.0  # sea-skimmer mean RCS
DEFAULT_DRONE_POS: tuple[float, float, float] = (-19990.0, 0.0, 50.0)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryData:
    """Parsed missile kinematic trajectory with drone-relative kinematics.

    The simulation output uses a 2-D plane (x, z).  We embed this
    into 3-D by setting ``y = 0`` everywhere, then compute range
    and radial velocity relative to the drone position.
    """
    filepath: Path
    header: dict[str, str]
    drone_pos: tuple[float, float, float]  # (x, y, z) of the radar drone
    # Raw kinematic arrays from the trajectory file.
    t_s: np.ndarray           # time (s)
    x_m: np.ndarray           # x position (m)
    z_m: np.ndarray           # altitude (m)
    vx_mps: np.ndarray        # x velocity (m/s)
    vz_mps: np.ndarray        # z velocity (m/s)
    mach: np.ndarray          # Mach number
    # Drone-relative quantities (computed at parse time).
    rel_range_m: np.ndarray   # Euclidean distance drone→target (m)
    rel_vr_mps: np.ndarray    # radial velocity (negative = closing)


@dataclass
class RadarTimeline:
    """Trajectory resampled to the radar CPI cadence."""
    t_cpi_s: np.ndarray       # CPI centre times (s)
    range_m: np.ndarray       # drone-relative range (m)
    v_radial_mps: np.ndarray  # radial velocity — signed (m/s)
    rcs_dbsm: np.ndarray      # Swerling-1 RCS per frame (dBsm)
    target_alt_m: np.ndarray  # target altitude per CPI frame (m)
    n_frames: int


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_missile_trajectory(
    filepath: Path,
    drone_pos: tuple[float, float, float] = DEFAULT_DRONE_POS,
) -> TrajectoryData:
    """Read a cruise-batch trajectory and compute drone-relative kinematics.

    The radar drone hovers at *drone_pos* ``(x, y, z)`` in the same
    coordinate frame as the missile.  The simulation is 2-D (x, z)
    so ``y`` is assumed zero for both drone and target.

    Relative quantities computed here:

    * **Relative range** — Euclidean distance from drone to target:
      ``R = sqrt(Δx² + Δy² + Δz²)``
    * **Radial velocity** — projection of the target velocity vector
      onto the line-of-sight (LOS) unit vector:
      ``v_r = (v⃗_tgt · r̂_LOS)``
      where ``r̂_LOS = (target − drone) / R``.
      Negative ``v_r`` means the target is *closing*; positive means
      it is *receding*.  When the missile passes directly beneath the
      drone, ``v_r`` crosses zero (Doppler notch).

    Parameters
    ----------
    filepath : Path
        Path to a trajectory ``.txt`` file.
    drone_pos : tuple[float, float, float]
        ``(x, y, z)`` position of the radar drone in metres.
        Default ``(-19990, 0, 50)`` — forward picket, 50 m altitude,
        10 m offset from the missile spawn at x = −20 000 m.

    Returns
    -------
    TrajectoryData
        Parsed trajectory with raw and drone-relative arrays.
    """
    lines = filepath.read_text(encoding="utf-8").splitlines()

    header: dict[str, str] = {}
    data_start_idx: int | None = None

    for i, line in enumerate(lines):
        # Header key : value pairs.
        if ":" in line and not line.strip().startswith(("=", "-")):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key:
                header[key] = val

        # Data begins after the dashed separator.
        if line.strip().startswith("---"):
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        raise ValueError(f"No data separator (---) found in {filepath}")

    # Parse numeric rows.
    rows: list[list[float]] = []
    for line in lines[data_start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            vals = [float(v) for v in stripped.split()]
            if len(vals) >= 6:
                rows.append(vals[:6])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No numeric data rows found in {filepath}")

    arr = np.array(rows, dtype=float)
    t_s   = arr[:, 0]
    x_m   = arr[:, 1]
    z_m   = arr[:, 2]
    vx    = arr[:, 3]
    vz    = arr[:, 4]
    mach  = arr[:, 5]

    # ── Drone-relative geometry ───────────────────────────────────
    # Target position vectors (y = 0 for the 2-D sim).
    dx = drone_pos[0]
    dy = drone_pos[1]
    dz = drone_pos[2]

    rel_x = x_m - dx
    rel_y = np.zeros_like(x_m)           # y_target − y_drone = 0
    rel_z = z_m - dz

    rel_range = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
    rel_range = np.maximum(rel_range, 1e-6)  # avoid division by zero

    # LOS unit vector components (target − drone direction).
    ux = rel_x / rel_range
    uy = rel_y / rel_range
    uz = rel_z / rel_range

    # Target velocity vector (vy = 0).
    # Radial velocity = dot(v_target, LOS_unit_vector).
    # Positive = target moving *away* from drone (receding).
    # Negative = target moving *toward* drone (closing).
    rel_vr = vx * ux + 0.0 * uy + vz * uz

    return TrajectoryData(
        filepath=filepath,
        header=header,
        drone_pos=drone_pos,
        t_s=t_s,
        x_m=x_m,
        z_m=z_m,
        vx_mps=vx,
        vz_mps=vz,
        mach=mach,
        rel_range_m=rel_range,
        rel_vr_mps=rel_vr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. CPI-Aligned Interpolation & Swerling-1 RCS
# ═══════════════════════════════════════════════════════════════════════════

def build_radar_timeline(
    traj: TrajectoryData,
    prf_hz: int = DEFAULT_PRF_HZ,
    cpi_pulses: int = DEFAULT_CPI_PULSES,
    base_rcs_dbsm: float = DEFAULT_BASE_RCS_DBSM,
    seed: int | None = None,
    swerling_model: int = 3,
) -> RadarTimeline:
    """Resample drone-relative trajectory onto the radar CPI cadence.

    Uses ``traj.rel_range_m`` and ``traj.rel_vr_mps`` (computed in
    ``parse_missile_trajectory`` relative to the drone position).

    Parameters
    ----------
    traj : TrajectoryData
        Parsed kinematic trajectory with drone-relative arrays.
    prf_hz : int
        Pulse repetition frequency (Hz).
    cpi_pulses : int
        Pulses per coherent processing interval.
    base_rcs_dbsm : float
        Mean RCS in dBsm for the Swerling fluctuation model.
    seed : int or None
        RNG seed for RCS scintillation.
    swerling_model : int
        1 → exponential RCS (many comparable scatterers); 3 →
        chi-squared 4-DOF (one dominant + small scatterers).
        Swerling 3 is the usual choice for anti-ship missiles
        (Kostis 2021, JDMS).  Fluctuation is drawn independently per
        CPI frame (fast/frame-to-frame decorrelation assumption).

    Returns
    -------
    RadarTimeline
        CPI-aligned timeline with fluctuating RCS.
    """
    cpi_duration_s = cpi_pulses / prf_hz  # e.g. 0.128 s
    t_max = float(traj.t_s[-1])
    n_frames = max(1, int(np.floor(t_max / cpi_duration_s)))

    # CPI centre times.
    t_cpi = np.arange(n_frames) * cpi_duration_s + cpi_duration_s / 2.0

    # Interpolate drone-relative range, radial velocity, and target altitude.
    range_interp = np.interp(t_cpi, traj.t_s, traj.rel_range_m)
    vr_interp    = np.interp(t_cpi, traj.t_s, traj.rel_vr_mps)
    alt_interp   = np.interp(t_cpi, traj.t_s, traj.z_m)

    # Swerling RCS scintillation.
    rng = np.random.default_rng(seed)
    base_rcs_linear = 10.0 ** (base_rcs_dbsm / 10.0)
    if swerling_model == 3:
        # Chi-squared, 4 DOF: Gamma(shape=2, scale=mean/2).
        rcs_linear = rng.gamma(
            shape=2.0, scale=base_rcs_linear / 2.0, size=n_frames
        )
    elif swerling_model == 1:
        rcs_linear = rng.exponential(scale=base_rcs_linear, size=n_frames)
    else:
        raise ValueError(f"swerling_model must be 1 or 3, got {swerling_model}")
    rcs_dbsm = 10.0 * np.log10(np.maximum(rcs_linear, 1e-30))

    return RadarTimeline(
        t_cpi_s=t_cpi,
        range_m=range_interp,
        v_radial_mps=vr_interp,
        rcs_dbsm=rcs_dbsm,
        target_alt_m=alt_interp,
        n_frames=n_frames,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Doppler Metadata (aliasing-aware)
# ═══════════════════════════════════════════════════════════════════════════

def compute_doppler_metadata(
    velocity_mps: float,
    lam: float,
    prf: int,
    cpi: int,
    bistatic_deg: float,
) -> dict[str, float | int]:
    """Compute Doppler metadata with correct PRF aliasing.

    Returns the *true* (unaliased) bistatic Doppler for physics
    labelling, the *aliased* Doppler that corresponds to the
    RD-map column after fftshift, the integer bin index into
    ``rd_video[:, :, bin]``, and the number of full PRF wraps.

    Parameters
    ----------
    velocity_mps : float
        Signed radial velocity in m/s.  Negative = closing (target
        approaching the drone) → positive Doppler; positive = receding
        → negative Doppler.  The sign must be preserved so closing and
        receding targets land on opposite sides of the Doppler axis
        (the Doppler-notch crossover as the missile passes the drone).
    lam : float
        Radar wavelength in metres.
    prf : int
        Pulse repetition frequency in Hz.
    cpi : int
        Number of pulses per coherent processing interval.
    bistatic_deg : float
        Bistatic angle β in degrees.

    Returns
    -------
    dict
        ``doppler_hz``         – true physical Doppler (Hz)
        ``doppler_hz_aliased`` – aliased into [-PRF/2, PRF/2) (Hz)
        ``doppler_bin``        – integer index into rd_video axis-2
        ``doppler_wraps``      – number of full PRF folds
    """
    cos_half = math.cos(math.radians(bistatic_deg / 2.0))

    # True physical Doppler — keep for physics labels.  The leading
    # minus maps closing (v_r < 0) to positive Doppler.
    fd_true = -2.0 * velocity_mps * cos_half / lam

    # Alias into [-PRF/2, PRF/2) — this is what the RD map shows.
    fd_res = prf / cpi
    fd_aliased = fd_true % prf
    if fd_aliased > prf / 2.0:
        fd_aliased -= prf

    # Integer bin index into rd_video axis=2 (after fftshift).
    doppler_bin = int(round(fd_aliased / fd_res)) % cpi

    return {
        "doppler_hz": fd_true,
        "doppler_hz_aliased": fd_aliased,
        "doppler_bin": doppler_bin,
        "doppler_wraps": int(abs(fd_true) // prf),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Frame-by-Frame RD Video Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_integrated_dataset(
    input_dir: str = "logs",
    output_dir: str = "integrated_output",
    drone_pos: tuple[float, float, float] = DEFAULT_DRONE_POS,
    bistatic_angle_deg: float = DEFAULT_BISTATIC_ANGLE_DEG,
    prf_hz: int = DEFAULT_PRF_HZ,
    cpi_pulses: int = DEFAULT_CPI_PULSES,
    n_range_bins: int = DEFAULT_N_RANGE_BINS,
    cnr_db: float = DEFAULT_CNR_DB,
    range_resolution_m: float = DEFAULT_RANGE_RES_M,
    base_rcs_dbsm: float = DEFAULT_BASE_RCS_DBSM,
    radar_params: RadarSystemParams | None = None,
    swerling_model: int = 3,
    seed: int | None = None,
) -> list[Path]:
    """Generate integrated Range-Doppler videos for all trajectories.

    For each ``.txt`` trajectory file in *input_dir*:

    1. Parse the kinematic data and resample onto CPI boundaries.
    2. For each CPI frame, generate K-distributed bistatic sea clutter,
       inject the target at the correct range bin and Doppler, add
       thermal noise, window, FFT, and store the power map (dB).
    3. Serialise the 3-D array ``(n_frames, n_range, n_doppler)`` and
       per-frame metadata into a compressed ``.npz`` file.

    Parameters
    ----------
    input_dir, output_dir : str
        Input trajectory folder and output folder.
    drone_pos : tuple[float, float, float]
        ``(x, y, z)`` position of the radar drone.  Default
        ``(-19990, 0, 50)`` — 10 m offset from missile spawn,
        50 m altitude (forward picket geometry).
    bistatic_angle_deg : float
        Bistatic angle β for clutter generation and Doppler correction.
    prf_hz, cpi_pulses, n_range_bins, cnr_db
        Radar / clutter parameters.
    range_resolution_m : float
        Range bin width in metres.
    base_rcs_dbsm : float
        Mean RCS in dBsm for the Swerling fluctuation model.
    radar_params : RadarSystemParams or None
        Radar system parameters for the range equation.  ``None``
        (default) uses ``RadarSystemParams()`` defaults — target SNR
        then falls off as R⁻⁴ instead of being range-independent.
    swerling_model : int
        RCS fluctuation model: 1 or 3 (default 3, per anti-ship
        missile convention).
    seed : int or None
        Master seed for reproducibility.  Per-trajectory seeds are
        derived from it and recorded in each file's radar_config.

    Returns
    -------
    list[Path]
        Paths to generated ``.npz`` files.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if radar_params is None:
        radar_params = RadarSystemParams(carrier_freq_ghz=77.0)

    seed_seq = np.random.SeedSequence(seed)

    txt_files = sorted(in_path.glob("cruise_run_*.txt"))
    if not txt_files:
        print(f"No trajectory files found in {in_path.resolve()}")
        return []

    print(f"Found {len(txt_files)} trajectory file(s) in {in_path.resolve()}")

    saved_files: list[Path] = []

    for traj_file in txt_files:
        stem = traj_file.stem
        print(f"\n{'─'*60}")
        print(f"Processing: {traj_file.name}")

        # Derive a recorded, reproducible seed for this trajectory.
        run_seed = int(seed_seq.spawn(1)[0].generate_state(1)[0])

        # ── parse & resample ──────────────────────────────────────
        traj = parse_missile_trajectory(traj_file, drone_pos=drone_pos)
        timeline = build_radar_timeline(
            traj,
            prf_hz=prf_hz,
            cpi_pulses=cpi_pulses,
            base_rcs_dbsm=base_rcs_dbsm,
            seed=run_seed,
            swerling_model=swerling_model,
        )

        n_frames = timeline.n_frames
        print(
            f"  Trajectory: {traj.t_s[-1]:.2f}s, "
            f"{len(traj.t_s)} kinematic steps → {n_frames} CPI frames"
        )

        # ── pre-allocate 3-D video array ──────────────────────────
        rd_video = np.zeros(
            (n_frames, n_range_bins, cpi_pulses), dtype=np.float32
        )
        frame_metadata: list[dict[str, Any]] = []

        rng = np.random.default_rng(run_seed + 1)

        # One model per trajectory: the Gamma texture must evolve
        # continuously across CPI frames (decorrelation ~3 s, frames
        # ~0.128 s apart), so the model carries texture state between
        # frames instead of being re-created per frame.
        model = BistaticClutterModel(
            bistatic_angle_deg=bistatic_angle_deg,
            carrier_freq_ghz=77.0,
            prf_hz=prf_hz,
            cpi_pulses=cpi_pulses,
            n_range_bins=n_range_bins,
            cnr_db=cnr_db,
            seed=run_seed,
        )
        cpi_duration_s = cpi_pulses / prf_hz

        for f_idx in trange(n_frames, desc=f"  {stem}", leave=False):
            r_m = float(timeline.range_m[f_idx])
            v_r = float(timeline.v_radial_mps[f_idx])
            rcs_db = float(timeline.rcs_dbsm[f_idx])
            tgt_alt = float(timeline.target_alt_m[f_idx])

            # Range bin for this frame.
            target_range_bin = int(r_m / range_resolution_m)

            # Doppler metadata (aliasing-aware, sign-preserving).
            dop = compute_doppler_metadata(
                velocity_mps=v_r,
                lam=LAMBDA_M,
                prf=prf_hz,
                cpi=cpi_pulses,
                bistatic_deg=bistatic_angle_deg,
            )
            doppler_hz = dop["doppler_hz"]
            doppler_hz_aliased = dop["doppler_hz_aliased"]
            doppler_bin = dop["doppler_bin"]
            doppler_wraps = dop["doppler_wraps"]

            # Track whether the target is in bounds.
            target_in_bounds = 0 <= target_range_bin < n_range_bins

            # ── generate clutter + target ─────────────────────────
            # First frame draws a fresh texture; subsequent frames
            # evolve it forward by one CPI duration.
            clutter_v = model.generate_clutter_voltage(
                dt_since_last_s=cpi_duration_s if f_idx > 0 else None
            )

            # Multipath propagation factor (Generalized Target model).
            # drone_pos[2] is the radar drone altitude; tgt_alt is the
            # interpolated missile altitude for this CPI frame.
            multipath_factor_db = 0.0
            if target_in_bounds:
                # Use the aliased Doppler for injection — this is what
                # the RD map FFT actually resolves.
                clutter_v = embed_target_in_clutter(
                    clutter_map=clutter_v,
                    target_range_bin=target_range_bin,
                    target_doppler_hz=doppler_hz_aliased,
                    target_rcs_dbsm=rcs_db,
                    noise_power=1.0,
                    prf_hz=prf_hz,
                    target_absolute_range_m=r_m,
                    drone_alt_m=drone_pos[2],
                    target_alt_m=tgt_alt,
                    radar_params=radar_params,
                )

                # Log the multipath factor magnitude for diagnostics.
                _F = multipath_propagation_factor(
                    target_range_m=r_m,
                    radar_alt_m=drone_pos[2],
                    target_alt_m=tgt_alt,
                    lambda_m=LAMBDA_M,
                )
                multipath_factor_db = float(
                    20.0 * np.log10(max(abs(_F), 1e-30))
                )

            # Thermal noise.
            noise = (
                rng.standard_normal(clutter_v.shape)
                + 1j * rng.standard_normal(clutter_v.shape)
            ) / np.sqrt(2.0)
            signal = clutter_v + noise

            # Hanning window → FFT → fftshift → power (dB).
            window = np.hanning(cpi_pulses)
            spectrum = np.fft.fftshift(
                np.fft.fft(signal * window[np.newaxis, :], axis=1),
                axes=1,
            )
            power_db = 10.0 * np.log10(
                np.maximum(np.abs(spectrum) ** 2, 1e-30)
            )

            rd_video[f_idx, :, :] = power_db.astype(np.float32)

            snr_single_pulse_db = float(
                10.0 * np.log10(
                    max(radar_params.snr_single_pulse(rcs_db, r_m), 1e-30)
                )
            )

            frame_metadata.append({
                "frame": f_idx,
                "t_s": float(timeline.t_cpi_s[f_idx]),
                "range_m": r_m,
                "snr_single_pulse_db": snr_single_pulse_db,
                "range_bin": target_range_bin if target_in_bounds else -1,
                "velocity_mps": v_r,
                "doppler_hz": doppler_hz,
                "doppler_hz_aliased": doppler_hz_aliased,
                "doppler_bin": doppler_bin,
                "doppler_wraps": doppler_wraps,
                "rcs_dbsm": rcs_db,
                "target_alt_m": tgt_alt,
                "multipath_factor_db": multipath_factor_db,
                "target_in_bounds": target_in_bounds,
            })

        # ── serialise ─────────────────────────────────────────────
        cos_half = math.cos(math.radians(bistatic_angle_deg / 2.0))
        radar_config = {
            "bistatic_angle_deg": bistatic_angle_deg,
            "prf_hz": prf_hz,
            "cpi_pulses": cpi_pulses,
            "n_range_bins": n_range_bins,
            "cnr_db": cnr_db,
            "range_resolution_m": range_resolution_m,
            "carrier_freq_ghz": 77.0,
            "lambda_m": LAMBDA_M,
            "base_rcs_dbsm": base_rcs_dbsm,
            "drone_pos": list(drone_pos),
            "doppler_bin_resolution_hz": prf_hz / cpi_pulses,
            "max_unambiguous_velocity_mps": (
                prf_hz * LAMBDA_M / (4.0 * cos_half)
            ),
            "velocity_folding": True,
            "swerling_model": swerling_model,
            "run_seed": run_seed,
            "radar_peak_power_w": radar_params.peak_power_w,
            "radar_tx_gain_db": radar_params.tx_gain_db,
            "radar_rx_gain_db": radar_params.rx_gain_db,
            "radar_noise_figure_db": radar_params.noise_figure_db,
            "radar_system_loss_db": radar_params.system_loss_db,
            "radar_bandwidth_hz": radar_params.bandwidth_hz,
        }

        out_file = out_path / f"{stem}.npz"
        np.savez_compressed(
            out_file,
            rd_video=rd_video,
            metadata=frame_metadata,
            radar_config=radar_config,
        )
        saved_files.append(out_file)

        in_bounds = sum(1 for m in frame_metadata if m["target_in_bounds"])
        print(
            f"  Saved: {out_file.name}  "
            f"({n_frames} frames, {in_bounds} with target in bounds)"
        )

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  INTEGRATION COMPLETE — {len(saved_files)} files in {out_path.resolve()}")
    print(f"{'═'*60}")
    for f in saved_files:
        sz_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name}  ({sz_mb:.1f} MB)")
    print()

    return saved_files


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate missile trajectories with bistatic sea clutter."
    )
    p.add_argument(
        "--input_dir", type=str, default="logs",
        help="Directory containing cruise_run_*.txt files.",
    )
    p.add_argument(
        "--output_dir", type=str, default="integrated_output",
        help="Directory to write .npz RD-video files.",
    )
    p.add_argument(
        "--bistatic_angle", type=float, default=DEFAULT_BISTATIC_ANGLE_DEG,
        help="Bistatic angle β in degrees (default 90).",
    )
    p.add_argument(
        "--prf", type=int, default=DEFAULT_PRF_HZ,
        help="Pulse repetition frequency in Hz (default 1000).",
    )
    p.add_argument(
        "--cpi", type=int, default=DEFAULT_CPI_PULSES,
        help="CPI pulses (default 128).",
    )
    p.add_argument(
        "--drone_pos", type=str, default="-19990,0,50",
        help="Drone (x,y,z) position as comma-separated floats (default -19990,0,50).",
    )
    p.add_argument(
        "--range_bins", type=int, default=DEFAULT_N_RANGE_BINS,
        help="Number of range bins (default 1500).",
    )
    p.add_argument(
        "--cnr", type=float, default=DEFAULT_CNR_DB,
        help="Clutter-to-noise ratio in dB (default 15).",
    )
    p.add_argument(
        "--range_res", type=float, default=DEFAULT_RANGE_RES_M,
        help="Range resolution in metres (default 3.0 → 4.5 km max range).",
    )
    p.add_argument(
        "--rcs", type=float, default=DEFAULT_BASE_RCS_DBSM,
        help="Mean RCS in dBsm (default -13).",
    )
    p.add_argument(
        "--swerling", type=int, default=3, choices=[1, 3],
        help="Swerling RCS fluctuation model (default 3).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Master seed for reproducible generation.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Parse drone position from comma-separated string.
    dp = tuple(float(x) for x in args.drone_pos.split(","))
    if len(dp) != 3:
        raise ValueError("--drone_pos must be 3 comma-separated floats: x,y,z")

    generate_integrated_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        drone_pos=dp,
        bistatic_angle_deg=args.bistatic_angle,
        prf_hz=args.prf,
        cpi_pulses=args.cpi,
        n_range_bins=args.range_bins,
        cnr_db=args.cnr,
        range_resolution_m=args.range_res,
        base_rcs_dbsm=args.rcs,
        swerling_model=args.swerling,
        seed=args.seed,
    )

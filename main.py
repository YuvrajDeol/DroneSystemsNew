from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

from src.guidance.guidance import SeaSkimGuidance
from src.output.data_logger import DataLogger
from src.physics.missile import Missile, MissileState


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _mach_to_mps(mach: float, speed_of_sound_mps: float) -> float:
    """Convert a Mach number to m/s."""
    return mach * speed_of_sound_mps


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg = _load_config(root / "config" / "sim_config.yaml")

    dt_s = float(cfg["sim"]["dt_s"])
    duration_s = float(cfg["sim"]["duration_s"])
    steps = int(np.ceil(duration_s / dt_s))
    speed_of_sound_mps = float(cfg["sim"]["speed_of_sound_mps"])

    # Target position — the missile flies *towards* x = 0 (target).
    t_pos_init = np.array(cfg["target"]["init_position_m"], dtype=float)
    t_vel = np.array(cfg["target"]["init_velocity_mps"], dtype=float)

    # Logs directory
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    num_runs = 20
    generated_files: list[str] = []

    for run_id in range(1, num_runs + 1):
        # ----------------------------------------------------------
        # Part 3 — Parameter randomization for this run
        # ----------------------------------------------------------
        cruise_altitude_m = random.uniform(4.5, 60.0)
        cruise_speed_mach = random.uniform(0.8, 3.5)
        sea_state = random.randint(0, 8)
        radar_altimeter_bias_m = random.uniform(-2.0, 2.0)
        air_temp_c = random.uniform(-20.0, 50.0)

        cruise_speed_mps = _mach_to_mps(cruise_speed_mach, speed_of_sound_mps)

        # ----------------------------------------------------------
        # Part 2 — Fresh state & log reset per run
        # ----------------------------------------------------------
        sim_log: list[dict] = []

        missile = Missile(
            mass_kg=float(cfg["missile"]["mass_kg"]),
            max_accel_mps2=float(cfg["missile"]["max_accel_mps2"]),
            air_density_kgpm3=float(cfg["missile"]["air_density_kgpm3"]),
            drag_area_m2=float(cfg["missile"]["drag_area_m2"]),
            cruise_speed_mps=cruise_speed_mps,
            speed_hold_kp=float(cfg["missile"]["speed_hold_kp"]),
            max_thrust_n=float(cfg["missile"]["max_thrust_n"]),
        )

        # Part 1 — Cruise-only guidance: disable terminal engagement
        # We pass cruise_only=True so the guidance never transitions
        # to popup/dive phases.
        guidance = SeaSkimGuidance(
            cruise_alt_m=cruise_altitude_m,
            terminal_range_m=float(cfg["guidance"]["terminal_range_m"]),
            popup_alt_m=float(cfg["guidance"]["popup_alt_m"]),
            max_accel_mps2=float(cfg["guidance"]["max_accel_mps2"]),
            max_vz_cmd_mps=float(cfg["guidance"]["max_vz_cmd_mps"]),
            alt_to_vz_gain=float(cfg["guidance"]["alt_to_vz_gain"]),
            vz_kp=float(cfg["guidance"]["vz_kp"]),
            vz_ki=float(cfg["guidance"]["vz_ki"]),
            vz_kd=float(cfg["guidance"]["vz_kd"]),
            N_gain=float(cfg["guidance"]["dive_nav_gain"]),
            cruise_only=True,  # <<< DISABLE terminal logic
        )

        # Missile initial state — apply radar altimeter bias to the
        # initial altitude perception so that the starting z is
        # offset from true cruise alt.
        init_pos = np.array(cfg["missile"]["init_position_m"], dtype=float)
        init_pos[1] = cruise_altitude_m + radar_altimeter_bias_m
        m_state = MissileState(
            position_m=init_pos,
            velocity_mps=np.array([cruise_speed_mps, 0.0], dtype=float),
        )
        t_pos = t_pos_init.copy()

        # Target distance for this run (missile must reach x = target x)
        target_x = t_pos[0]

        terminated_reason = "max_duration"

        # ----------------------------------------------------------
        # Simulation loop (cruise only)
        # ----------------------------------------------------------
        for k in range(steps):
            t_s = k * dt_s

            a_cmd = guidance.accel_command(
                missile_pos_m=m_state.position_m,
                missile_vel_mps=m_state.velocity_mps,
                target_pos_m=t_pos,
                target_vel_mps=t_vel,
                dt_s=dt_s,
            )

            # Part 1 — Continuous cruise: always hold cruise speed
            m_state = missile.step(
                m_state, a_cmd, dt_s, hold_cruise_speed=True
            )

            # Clamp sea level
            position_m = m_state.position_m.copy()
            velocity_mps = m_state.velocity_mps.copy()

            # Part 1 — Impact / sea-ditch condition (altitude <= 0)
            if position_m[1] <= 0.0:
                position_m[1] = 0.0
                velocity_mps[1] = max(0.0, velocity_mps[1])
                m_state = MissileState(
                    position_m=position_m,
                    velocity_mps=velocity_mps,
                )
                mach = float(np.linalg.norm(m_state.velocity_mps) / speed_of_sound_mps)
                sim_log.append({
                    "t_s": t_s,
                    "x_m": m_state.position_m[0],
                    "z_m": m_state.position_m[1],
                    "vx": m_state.velocity_mps[0],
                    "vz": m_state.velocity_mps[1],
                    "mach": mach,
                })
                terminated_reason = "sea_ditch"
                break

            m_state = MissileState(
                position_m=position_m,
                velocity_mps=velocity_mps,
            )

            t_pos = t_pos + t_vel * dt_s

            mach = float(np.linalg.norm(m_state.velocity_mps) / speed_of_sound_mps)
            sim_log.append({
                "t_s": t_s,
                "x_m": m_state.position_m[0],
                "z_m": m_state.position_m[1],
                "vx": m_state.velocity_mps[0],
                "vz": m_state.velocity_mps[1],
                "mach": mach,
            })

            # Part 1 — End when the missile has reached the target distance
            if m_state.position_m[0] >= target_x:
                terminated_reason = "reached_target"
                break

        # ----------------------------------------------------------
        # Part 4 — Write per-run .txt log with unique filename
        # ----------------------------------------------------------
        filename = (
            f"cruise_run_{run_id:02d}_Alt_{cruise_altitude_m:.1f}m"
            f"_Mach_{cruise_speed_mach:.2f}.txt"
        )
        filepath = logs_dir / filename
        generated_files.append(filename)

        with filepath.open("w", encoding="utf-8") as f:
            # Log header with randomized parameters
            f.write("=" * 72 + "\n")
            f.write(f"  CRUISE-ONLY BATCH RUN #{run_id:02d}\n")
            f.write("=" * 72 + "\n")
            f.write(f"  cruise_altitude_m      : {cruise_altitude_m:.2f}\n")
            f.write(f"  cruise_speed_mach      : {cruise_speed_mach:.4f}\n")
            f.write(f"  cruise_speed_mps       : {cruise_speed_mps:.2f}\n")
            f.write(f"  sea_state              : {sea_state}\n")
            f.write(f"  radar_altimeter_bias_m : {radar_altimeter_bias_m:.4f}\n")
            f.write(f"  air_temp_c             : {air_temp_c:.2f}\n")
            f.write(f"  termination            : {terminated_reason}\n")
            f.write(f"  total_steps            : {len(sim_log)}\n")
            f.write("=" * 72 + "\n\n")

            # Column header
            f.write(
                f"{'t_s':>10s}  {'x_m':>14s}  {'z_m':>10s}  "
                f"{'vx':>12s}  {'vz':>10s}  {'mach':>8s}\n"
            )
            f.write("-" * 72 + "\n")

            for row in sim_log:
                f.write(
                    f"{row['t_s']:10.4f}  {row['x_m']:14.4f}  {row['z_m']:10.4f}  "
                    f"{row['vx']:12.4f}  {row['vz']:10.4f}  {row['mach']:8.4f}\n"
                )

        print(
            f"[Run {run_id:02d}/{num_runs}] {terminated_reason:>15s} | "
            f"Alt={cruise_altitude_m:6.1f}m  Mach={cruise_speed_mach:.2f}  "
            f"SeaSt={sea_state}  Bias={radar_altimeter_bias_m:+.2f}m  "
            f"Temp={air_temp_c:+.1f}°C  =>  {filename}"
        )

    # ----------------------------------------------------------
    # Part 4.9 — Completion summary
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  BATCH COMPLETE — {len(generated_files)} files generated in logs/")
    print("=" * 72)
    for fn in generated_files:
        print(f"  • {fn}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

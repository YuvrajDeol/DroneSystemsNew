from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.guidance.guidance import SeaSkimGuidance
from src.output.data_logger import DataLogger
from src.physics.missile import Missile, MissileState


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    root = Path(__file__).resolve().parent
    cfg = _load_config(root / "config" / "sim_config.yaml")

    dt_s = float(cfg["sim"]["dt_s"])
    duration_s = float(cfg["sim"]["duration_s"])
    steps = int(np.ceil(duration_s / dt_s))

    missile = Missile(
        mass_kg=float(cfg["missile"]["mass_kg"]),
        max_accel_mps2=float(cfg["missile"]["max_accel_mps2"]),
        air_density_kgpm3=float(cfg["missile"]["air_density_kgpm3"]),
        drag_area_m2=float(cfg["missile"]["drag_area_m2"]),
        cruise_speed_mps=float(cfg["missile"]["cruise_speed_mps"]),
        speed_hold_kp=float(cfg["missile"]["speed_hold_kp"]),
        max_thrust_n=float(cfg["missile"]["max_thrust_n"]),
    )
    guidance = SeaSkimGuidance(
        cruise_alt_m=float(cfg["guidance"]["cruise_alt_m"]),
        terminal_range_m=float(cfg["guidance"]["terminal_range_m"]),
        popup_alt_m=float(cfg["guidance"]["popup_alt_m"]),
        max_accel_mps2=float(cfg["guidance"]["max_accel_mps2"]),
        max_vz_cmd_mps=float(cfg["guidance"]["max_vz_cmd_mps"]),
        alt_to_vz_gain=float(cfg["guidance"]["alt_to_vz_gain"]),
        vz_kp=float(cfg["guidance"]["vz_kp"]),
        vz_ki=float(cfg["guidance"]["vz_ki"]),
        vz_kd=float(cfg["guidance"]["vz_kd"]),
        dive_nav_gain=float(cfg["guidance"]["dive_nav_gain"]),
    )

    m_state = MissileState(
        position_m=np.array(cfg["missile"]["init_position_m"], dtype=float),
        velocity_mps=np.array(cfg["missile"]["init_velocity_mps"], dtype=float),
    )
    t_pos = np.array(cfg["target"]["init_position_m"], dtype=float)
    t_vel = np.array(cfg["target"]["init_velocity_mps"], dtype=float)
    speed_of_sound_mps = float(cfg["sim"]["speed_of_sound_mps"])

    logger = DataLogger(out_dir=(root / cfg["output"]["out_dir"]).resolve())

    for k in range(steps):
        t_s = k * dt_s

        a_cmd = guidance.accel_command(
            missile_pos_m=m_state.position_m,
            missile_vel_mps=m_state.velocity_mps,
            target_pos_m=t_pos,
            target_vel_mps=t_vel,
            dt_s=dt_s,
        )

        m_state = missile.step(
            m_state, a_cmd, dt_s, hold_cruise_speed=(guidance.phase == "cruise")
        )
        # Keep sea level at z=0 and altitude AGL non-negative.
        m_state = MissileState(
            position_m=np.array(
                [m_state.position_m[0], max(0.0, m_state.position_m[1])], dtype=float
            ),
            velocity_mps=m_state.velocity_mps,
        )
        t_pos = t_pos + t_vel * dt_s

        miss_distance_m = float(np.linalg.norm(t_pos - m_state.position_m))
        mach = float(np.linalg.norm(m_state.velocity_mps) / speed_of_sound_mps)

        logger.log(
            t_s=t_s,
            x_m=m_state.position_m[0],
            z_m=m_state.position_m[1],
            vx=m_state.velocity_mps[0],
            vz=m_state.velocity_mps[1],
            mach=mach,
            miss_distance_m=miss_distance_m,
        )

        if miss_distance_m < 5.0:
            break

    out_path = logger.to_csv(filename=str(cfg["output"]["csv_name"]))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


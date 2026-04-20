from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SeaSkimGuidance:
    """Sea-skimming profile controller in x-z plane.

    Phases:
    - cruise: hold low altitude using radar-altimeter-driven vz control
    - popup: climb to a commanded pop-up altitude
    - dive:   command towards target for terminal attack
    """

    cruise_alt_m: float = 5.0
    terminal_range_m: float = 5000.0
    popup_alt_m: float = 150.0
    max_accel_mps2: float = 30.0
    max_vz_cmd_mps: float = 25.0
    alt_to_vz_gain: float = 0.6
    vz_kp: float = 2.4
    vz_ki: float = 0.35
    vz_kd: float = 0.4
    dive_nav_gain: float = 1.2

    phase: str = "cruise"
    _vz_err_int: float = 0.0
    _prev_vz_err: float = 0.0

    def _vz_pid(self, vz_target_mps: float, vz_mps: float, dt_s: float) -> float:
        vz_err = vz_target_mps - vz_mps
        self._vz_err_int += vz_err * dt_s
        self._vz_err_int = float(np.clip(self._vz_err_int, -50.0, 50.0))
        vz_err_dot = (vz_err - self._prev_vz_err) / max(1e-6, dt_s)
        self._prev_vz_err = vz_err
        return (
            self.vz_kp * vz_err
            + self.vz_ki * self._vz_err_int
            + self.vz_kd * vz_err_dot
        )

    def accel_command(
        self,
        missile_pos_m: np.ndarray,
        missile_vel_mps: np.ndarray,
        target_pos_m: np.ndarray,
        target_vel_mps: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        del target_vel_mps
        m_p = np.asarray(missile_pos_m, dtype=float).reshape(2)  # [x, z]
        m_v = np.asarray(missile_vel_mps, dtype=float).reshape(2)  # [vx, vz]
        t_p = np.asarray(target_pos_m, dtype=float).reshape(2)  # [x, z]

        rel = t_p - m_p
        range_m = float(np.linalg.norm(rel))

        if self.phase == "cruise" and range_m < self.terminal_range_m:
            self.phase = "popup"
            self._vz_err_int = 0.0
            self._prev_vz_err = 0.0

        if self.phase == "popup" and m_p[1] >= self.popup_alt_m - 0.1:
            self.phase = "dive"
            self._vz_err_int = 0.0
            self._prev_vz_err = 0.0

        if self.phase == "cruise":
            alt_err_m = self.cruise_alt_m - m_p[1]
            vz_target = float(
                np.clip(
                    self.alt_to_vz_gain * alt_err_m,
                    -self.max_vz_cmd_mps,
                    self.max_vz_cmd_mps,
                )
            )
            az_cmd = self._vz_pid(vz_target, m_v[1], dt_s)
            ax_cmd = 0.0
        elif self.phase == "popup":
            alt_err_m = self.popup_alt_m - m_p[1]
            vz_target = float(
                np.clip(
                    self.alt_to_vz_gain * alt_err_m,
                    -self.max_vz_cmd_mps,
                    self.max_vz_cmd_mps,
                )
            )
            az_cmd = self._vz_pid(vz_target, m_v[1], dt_s)
            ax_cmd = 0.0
        else:  # dive
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm < 1e-6:
                return np.zeros(2, dtype=float)
            rel_hat = rel / rel_norm
            desired_vel = rel_hat * max(50.0, float(np.linalg.norm(m_v)))
            vel_err = desired_vel - m_v
            accel_vec = self.dive_nav_gain * vel_err
            ax_cmd = float(accel_vec[0])
            az_cmd = float(accel_vec[1])

        a_cmd = np.array([ax_cmd, az_cmd], dtype=float)
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm > self.max_accel_mps2:
            a_cmd *= self.max_accel_mps2 / a_norm
        return a_cmd


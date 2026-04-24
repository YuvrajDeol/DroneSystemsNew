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
    terminal_range_m: float = 15000.0
    popup_alt_m: float = 150.0
    max_accel_mps2: float = 30.0
    max_vz_cmd_mps: float = 25.0
    alt_to_vz_gain: float = 0.6
    vz_kp: float = 2.4
    vz_ki: float = 0.35
    vz_kd: float = 0.4
    N_gain: float = 3.0
    cruise_only: bool = False

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
        m_p = np.asarray(missile_pos_m, dtype=float).reshape(2)  # [x, z]
        m_v = np.asarray(missile_vel_mps, dtype=float).reshape(2)  # [vx, vz]
        t_p = np.asarray(target_pos_m, dtype=float).reshape(2)  # [x, z]
        t_v = np.asarray(target_vel_mps, dtype=float).reshape(2)  # [vx, vz]

        rel = t_p - m_p
        range_m = float(np.linalg.norm(rel))

        # --- Terminal phase transitions (disabled when cruise_only=True) ---
        if not self.cruise_only:
            if self.phase == "cruise" and range_m < self.terminal_range_m:
                self.phase = "popup"
                self._vz_err_int = 0.0
                self._prev_vz_err = 0.0

            # TODO: Implement 3-State Kalman Filter for Radar Altimeter wave noise rejection.
            if self.phase == "popup" and m_p[1] >= self.popup_alt_m - 0.1:
                self.phase = "dive"
                self._vz_err_int = 0.0
                self._prev_vz_err = 0.0

        if self.phase == "cruise":
            # TODO: Implement 3-State Kalman Filter for Radar Altimeter wave noise rejection.
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
            # TODO: Implement 3-State Kalman Filter for Radar Altimeter wave noise rejection.
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
            rel_p = t_p - m_p
            rel_v = t_v - m_v
            R = float(np.linalg.norm(rel_p))
            if R < 1e-6:
                return np.zeros(2, dtype=float)

            V_c = -float(np.dot(rel_p, rel_v)) / R
            lambda_dot = (rel_p[0] * rel_v[1] - rel_p[1] * rel_v[0]) / max(R * R, 1e-6)
            a_n = self.N_gain * V_c * lambda_dot

            speed = float(np.linalg.norm(m_v))
            if speed < 1e-6:
                return np.zeros(2, dtype=float)
            n_hat = np.array([-m_v[1], m_v[0]], dtype=float) / speed
            accel_vec = a_n * n_hat
            ax_cmd = float(accel_vec[0])
            az_cmd = float(accel_vec[1])

        a_cmd = np.array([ax_cmd, az_cmd], dtype=float)
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm > self.max_accel_mps2:
            a_cmd *= self.max_accel_mps2 / a_norm
        return a_cmd


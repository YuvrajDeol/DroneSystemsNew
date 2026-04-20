from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MissileState:
    """2D kinematic state in an ENU-like plane.

    Units:
    - position: meters
    - velocity: meters/second
    """

    position_m: np.ndarray  # shape (2,)
    velocity_mps: np.ndarray  # shape (2,)


@dataclass(slots=True)
class Missile:
    mass_kg: float = 100.0
    max_accel_mps2: float = 30.0
    air_density_kgpm3: float = 1.225
    drag_area_m2: float = 0.05  # effective Cd * A
    cruise_speed_mps: float = 850.0
    speed_hold_kp: float = 0.03
    max_thrust_n: float = 12000.0

    def step(
        self,
        state: MissileState,
        accel_cmd_mps2: np.ndarray,
        dt_s: float,
        hold_cruise_speed: bool = False,
    ) -> MissileState:
        accel = np.asarray(accel_cmd_mps2, dtype=float).reshape(2).copy()

        vel = state.velocity_mps
        speed = float(np.linalg.norm(vel))
        if speed > 1e-6:
            vel_hat = vel / speed
        else:
            vel_hat = np.array([1.0, 0.0], dtype=float)

        # Axial drag model: D = 0.5 * rho * v^2 * CdA.
        drag_n = 0.5 * self.air_density_kgpm3 * speed * speed * self.drag_area_m2
        drag_accel_vec = -(drag_n / self.mass_kg) * vel_hat

        thrust_n = 0.0
        if hold_cruise_speed:
            speed_error = self.cruise_speed_mps - speed
            thrust_cmd = drag_n + self.speed_hold_kp * self.mass_kg * speed_error
            thrust_n = float(np.clip(thrust_cmd, 0.0, self.max_thrust_n))
        thrust_accel_vec = (thrust_n / self.mass_kg) * vel_hat

        accel += drag_accel_vec + thrust_accel_vec

        a_norm = float(np.linalg.norm(accel))
        if a_norm > self.max_accel_mps2:
            accel *= self.max_accel_mps2 / a_norm

        v_next = state.velocity_mps + accel * dt_s
        p_next = state.position_m + v_next * dt_s
        return MissileState(position_m=p_next, velocity_mps=v_next)


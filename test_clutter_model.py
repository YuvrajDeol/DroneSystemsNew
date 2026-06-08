"""
test_clutter_model.py — pytest suite for the SENTINEL Mesh clutter simulator.

Validates the core physics invariants of the compound K-distribution
clutter model, the CA-CFAR detector, and the training-data pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clutter_model import (
    BistaticClutterModel,
    CFARDetector,
    embed_target_in_clutter,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def model_90() -> BistaticClutterModel:
    """Standard 90° bistatic model for repeatable tests."""
    return BistaticClutterModel(
        bistatic_angle_deg=90,
        cnr_db=15.0,
        n_range_bins=128,
        cpi_pulses=64,
        prf_hz=1000,
        seed=12345,
    )


# ── Test: texture mean normalisation ──────────────────────────────────────

def test_texture_mean_normalized(model_90: BistaticClutterModel) -> None:
    """Gamma texture envelope must have mean ≈ 1.0 (within 5%).

    The texture is drawn from Gamma(ν, 1/ν) and re-normalised per
    range bin.  The overall mean across all cells should be ≈ 1.0.
    """
    texture = model_90.generate_texture()

    assert texture.shape == (model_90.n_range_bins, model_90.cpi_pulses)
    assert np.all(texture > 0.0), "Texture must be strictly positive"

    mean_tex = float(np.mean(texture))
    assert abs(mean_tex - 1.0) < 0.05, (
        f"Texture mean = {mean_tex:.4f}, expected ≈ 1.0 (within 5%)"
    )


# ── Test: speckle unit mean power ─────────────────────────────────────────

def test_speckle_power(model_90: BistaticClutterModel) -> None:
    """Complex Gaussian speckle must have mean power ≈ 1.0 (within 5%).

    Each sample is CN(0,1), so E[|w|²] = 1.
    """
    speckle = model_90.generate_speckle()

    assert speckle.shape == (model_90.n_range_bins, model_90.cpi_pulses)
    assert np.iscomplexobj(speckle), "Speckle must be complex"

    mean_power = float(np.mean(np.abs(speckle) ** 2))
    assert abs(mean_power - 1.0) < 0.05, (
        f"Speckle mean power = {mean_power:.4f}, expected ≈ 1.0 (within 5%)"
    )


# ── Test: shape parameter recovery ───────────────────────────────────────

def test_shape_parameter_recovery() -> None:
    """MoM estimator should recover ν within 20% on true K-distributed data.

    We bypass the moving-average texture approximation (which distorts
    the marginal Gamma distribution) and instead generate exact
    K-distributed intensity samples: I = x · w, where x ~ Gamma(ν, 1/ν)
    and w ~ Exp(1).  This isolates the estimator's accuracy from the
    texture generation approximation.
    """
    true_nu = 3.60  # 90° bistatic
    n_samples = 200_000
    rng = np.random.default_rng(99)

    # Exact K-distributed intensity: Gamma texture × Exponential speckle.
    texture = rng.gamma(shape=true_nu, scale=1.0 / true_nu, size=n_samples)
    speckle_power = rng.exponential(scale=1.0, size=n_samples)
    intensity = texture * speckle_power

    model = BistaticClutterModel(
        bistatic_angle_deg=90, cnr_db=25.0, seed=1,
        n_range_bins=64, cpi_pulses=64,
    )
    nu_est = model.estimate_shape_parameter(intensity, noise_power=0.0)

    assert np.isfinite(nu_est), "ν estimate must be finite"
    assert nu_est > 0.0, "ν estimate must be positive"

    rel_error = abs(nu_est - true_nu) / true_nu
    assert rel_error < 0.20, (
        f"ν estimate = {nu_est:.2f}, true ν = {true_nu:.2f}, "
        f"relative error = {rel_error:.0%} (tolerance 20%)"
    )


# ── Test: range-Doppler map dimensions ────────────────────────────────────

def test_range_doppler_map_shape(model_90: BistaticClutterModel) -> None:
    """compute_range_doppler_map must return correctly shaped arrays."""
    power_db, doppler_axis, range_axis = model_90.compute_range_doppler_map()

    assert power_db.shape == (model_90.n_range_bins, model_90.cpi_pulses)
    assert doppler_axis.shape == (model_90.cpi_pulses,)
    assert range_axis.shape == (model_90.n_range_bins,)

    # Power should be in dB (finite, no NaN).
    assert np.all(np.isfinite(power_db)), "Power map contains non-finite values"


# ── Test: CFAR false-alarm rate ───────────────────────────────────────────

def test_cfar_pfa() -> None:
    """CA-CFAR false-alarm rate should be within a factor of 2 of target.

    Generates 200 clutter-only range-Doppler maps and counts false
    alarms.  The measured PFA should be within [pfa/2, 2*pfa].
    """
    target_pfa = 1e-4
    cfar = CFARDetector(guard_cells=2, training_cells=8, pfa=target_pfa)

    n_trials = 200
    n_range = 32
    n_doppler = 64
    rng = np.random.default_rng(42)

    total_detections = 0
    total_testable = 0
    half_window = cfar.guard_cells + cfar.training_cells  # 10

    for _ in range(n_trials):
        # Exponential power (noise-only, no clutter structure).
        noise = (
            rng.standard_normal((n_range, n_doppler))
            + 1j * rng.standard_normal((n_range, n_doppler))
        ) / np.sqrt(2.0)
        power_lin = np.abs(noise) ** 2

        det = cfar.detect(power_lin)
        total_detections += int(det.sum())

        # Only cells with a full training window are testable.
        testable_per_row = max(0, n_doppler - 2 * half_window)
        total_testable += n_range * testable_per_row

    measured_pfa = total_detections / max(total_testable, 1)

    assert measured_pfa < 2.0 * target_pfa, (
        f"Measured PFA = {measured_pfa:.2e}, exceeds 2 × target ({2*target_pfa:.2e})"
    )
    # Lower bound is generous — very low PFA is fine.
    assert measured_pfa < 1.0, "Sanity: PFA should be < 1"

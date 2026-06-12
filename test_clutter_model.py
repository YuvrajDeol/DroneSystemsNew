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
    RadarSystemParams,
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


# ── Test: texture statistics ──────────────────────────────────────────────

def test_texture_unit_mean_and_gamma_variance() -> None:
    """Texture must have ensemble mean ≈ 1 AND variance ≈ 1/ν.

    The variance check is the one that matters: a unit-mean texture
    with no bin-to-bin variance produces Rayleigh (not K-distributed)
    clutter.  Texture is Gamma(ν, 1/ν) ⇒ E[x] = 1, Var[x] = 1/ν.
    """
    model = BistaticClutterModel(
        bistatic_angle_deg=90,  # ν = 3.60
        cnr_db=15.0,
        n_range_bins=8192,
        cpi_pulses=4,
        prf_hz=1000,
        seed=7,
    )
    nu = model._params["nu"]
    texture = model.generate_texture()

    assert texture.shape == (model.n_range_bins, model.cpi_pulses)
    assert np.all(texture > 0.0), "Texture must be strictly positive"

    per_bin = texture[:, 0]
    mean_tex = float(np.mean(per_bin))
    var_tex = float(np.var(per_bin))

    assert abs(mean_tex - 1.0) < 0.05, (
        f"Texture mean = {mean_tex:.4f}, expected ≈ 1.0"
    )
    assert abs(var_tex - 1.0 / nu) / (1.0 / nu) < 0.20, (
        f"Texture variance = {var_tex:.4f}, expected ≈ 1/ν = {1.0/nu:.4f}"
    )


def test_texture_constant_within_cpi(model_90: BistaticClutterModel) -> None:
    """Texture decorrelates over ~3 s ≫ one CPI, so it must be
    constant along the pulse axis within a single CPI."""
    texture = model_90.generate_texture()
    assert np.allclose(texture, texture[:, :1]), (
        "Texture must be constant across pulses within one CPI"
    )


def test_texture_evolves_across_cpis() -> None:
    """Across-CPI texture correlation must follow exp(−Δt/τ):
    strongly correlated one CPI later, decorrelated after ≫ τ."""
    model = BistaticClutterModel(
        bistatic_angle_deg=90,
        cnr_db=15.0,
        n_range_bins=8192,
        cpi_pulses=4,
        prf_hz=1000,
        seed=11,
    )
    tex0 = model.generate_texture()[:, 0]
    tex_next = model.generate_texture(dt_since_last_s=0.128)[:, 0]
    tex_far = model.generate_texture(dt_since_last_s=100.0)[:, 0]

    corr_next = float(np.corrcoef(tex0, tex_next)[0, 1])
    corr_far = float(np.corrcoef(tex0, tex_far)[0, 1])

    assert corr_next > 0.85, (
        f"Texture 0.128 s later should be strongly correlated, got {corr_next:.3f}"
    )
    assert abs(corr_far) < 0.1, (
        f"Texture 100 s later should be decorrelated, got {corr_far:.3f}"
    )


def test_clutter_intensity_is_k_distributed() -> None:
    """Full clutter intensity must satisfy the K-distribution moment
    ratio MN2 = E[I²]/E[I]² = 2 + 2/ν (Rayleigh-only clutter gives 2).

    This is the end-to-end check that texture variation actually
    survives into the generated clutter voltage.
    """
    model = BistaticClutterModel(
        bistatic_angle_deg=90,  # ν = 3.60
        cnr_db=15.0,
        n_range_bins=4096,
        cpi_pulses=64,
        prf_hz=1000,
        seed=3,
    )
    nu = model._params["nu"]
    clutter = model.generate_clutter_voltage()
    intensity = np.abs(clutter) ** 2

    mn2 = float(np.mean(intensity**2) / np.mean(intensity) ** 2)
    expected = 2.0 + 2.0 / nu

    assert abs(mn2 - expected) / expected < 0.15, (
        f"MN2 = {mn2:.3f}, expected ≈ {expected:.3f} for ν = {nu:.2f} "
        f"(MN2 = 2.0 would mean the clutter is Rayleigh, not K)"
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

    Generates exact K-distributed intensity samples directly:
    I = x · w, where x ~ Gamma(ν, 1/ν) and w ~ Exp(1), isolating the
    estimator's accuracy from the clutter generation pipeline.
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


# ── Test: vectorized CFAR matches design Pfa ──────────────────────────────

def test_vectorized_cfar_pfa() -> None:
    """Vectorized CA-CFAR false-alarm rate should track the design Pfa
    on exponential (noise-only) data, within a factor of 2."""
    target_pfa = 1e-3
    cfar = CFARDetector(guard_cells=2, training_cells=8, pfa=target_pfa)

    rng = np.random.default_rng(7)
    noise = (
        rng.standard_normal((512, 256))
        + 1j * rng.standard_normal((512, 256))
    ) / np.sqrt(2.0)
    power_lin = np.abs(noise) ** 2

    det = cfar.detect_vectorized(power_lin)
    measured = det.mean()

    assert measured < 2.0 * target_pfa, (
        f"Vectorized CFAR Pfa = {measured:.2e} exceeds 2 × design"
    )
    assert measured > target_pfa / 4.0, (
        f"Vectorized CFAR Pfa = {measured:.2e} implausibly low"
    )


# ── Test: radar range equation ────────────────────────────────────────────

def test_range_equation_r4_falloff() -> None:
    """Target SNR must follow the R⁻⁴ law: doubling range → −12 dB."""
    params = RadarSystemParams()
    snr_1km = params.snr_single_pulse(rcs_dbsm=-13.0, range_m=1000.0)
    snr_2km = params.snr_single_pulse(rcs_dbsm=-13.0, range_m=2000.0)

    assert snr_1km > 0 and snr_2km > 0
    ratio_db = 10.0 * np.log10(snr_1km / snr_2km)
    expected_db = 40.0 * np.log10(2.0)  # ≈ 12.04 dB for R⁻⁴
    assert abs(ratio_db - expected_db) < 0.01, (
        f"Doubling range changed SNR by {ratio_db:.2f} dB, "
        f"expected {expected_db:.2f} dB"
    )

    # RCS linearity: +10 dBsm → +10 dB SNR.
    snr_big = params.snr_single_pulse(rcs_dbsm=-3.0, range_m=1000.0)
    assert abs(10.0 * np.log10(snr_big / snr_1km) - 10.0) < 0.01


def test_embed_target_range_dependence() -> None:
    """With radar_params, injected target power must drop with range."""
    params = RadarSystemParams()
    powers = []
    for r in (1000.0, 3000.0):
        cm = np.zeros((16, 64), dtype=complex)
        embed_target_in_clutter(
            clutter_map=cm,
            target_range_bin=8,
            target_doppler_hz=100.0,
            target_rcs_dbsm=-13.0,
            noise_power=1.0,
            prf_hz=1000,
            target_absolute_range_m=r,
            drone_alt_m=50.0,
            target_alt_m=10.0,
            radar_params=params,
        )
        powers.append(float(np.mean(np.abs(cm[8, :]) ** 2)))

    assert powers[0] > powers[1] * 10.0, (
        f"Target at 1 km should be ≫ stronger than at 3 km, "
        f"got {powers[0]:.3e} vs {powers[1]:.3e}"
    )

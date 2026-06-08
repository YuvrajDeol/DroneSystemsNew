"""
clutter_model.py — Compound K-Distribution Bistatic Sea Clutter Simulator
=========================================================================

Physically-grounded 77 GHz bistatic sea clutter model for SENTINEL Mesh
radar training data generation.  Based on the NetRAD empirical dataset
(Ritchie et al., S-band 2.45 GHz bistatic sea clutter) with all
parameters frequency-scaled to 77 GHz.

References
----------
- M. Ritchie et al., "Bistatic radar sea clutter statistics,"
  IET Radar Sonar & Navigation, 2015.
- NetRAD S-band (2.45 GHz) bistatic sea clutter campaign.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import special

# ═══════════════════════════════════════════════════════════════════════════
# KNOWN LIMITATIONS
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_LIMITATIONS: str = """\
- ν values are from sea state 3 (H_1/3 ≈ 3.3–4.0 m). Calmer seas → lower ν.
- NetRAD grazing angles were 0.6°–1.5°. Higher drone altitude → higher
  grazing angle → different ν.
- NetRAD range resolution was 4.9 m. At mmWave with wider bandwidth, finer
  resolution cells contain fewer scatterers → spikier clutter → smaller ν.
  The tabulated ν values are optimistic upper bounds.
- B gradient values are approximated; exact values require reading Figures
  9/10 of Ritchie et al. directly.
- Bistatic Doppler correction uses cos(β/2) approximation (broadside
  geometry).
- VV polarization parameters are available but not primary for SENTINEL
  Mesh HH config.
"""

# ═══════════════════════════════════════════════════════════════════════════
# EMPIRICAL CONSTANTS (NetRAD, HH polarisation)
# ═══════════════════════════════════════════════════════════════════════════

NETRAD_PARAMS_HH: dict[int, dict[str, float]] = {
    60:  {"nu": 0.32, "sigma_psd_hz_sband": 13.54, "cog_hz_sband":  8.20},
    90:  {"nu": 3.60, "sigma_psd_hz_sband": 12.72, "cog_hz_sband":  8.39},
    120: {"nu": 5.90, "sigma_psd_hz_sband": 10.90, "cog_hz_sband":  1.20},
}

FREQ_SCALE: float = 77.0 / 2.45          # ≈ 31.43 — S-band → 77 GHz multiplier
SPECKLE_DECORR_MS: float = 40.0 * (2.45 / 77.0)  # ≈ 1.27 ms at 77 GHz

# Anchor angles used for interpolation (sorted).
_ANCHOR_ANGLES = sorted(NETRAD_PARAMS_HH.keys())  # [60, 90, 120]


# ═══════════════════════════════════════════════════════════════════════════
# BistaticClutterModel
# ═══════════════════════════════════════════════════════════════════════════

class BistaticClutterModel:
    """Compound K-distribution bistatic sea clutter generator.

    Produces physically realistic range-Doppler clutter frames at 77 GHz
    using the compound K-distribution model (Gamma texture × Rayleigh
    speckle), parameterised from the NetRAD S-band empirical dataset and
    frequency-scaled to mmWave.

    Parameters
    ----------
    bistatic_angle_deg : float
        Bistatic angle β in degrees.  Must be within [60, 120].
        Values between anchor points (60, 90, 120) are linearly
        interpolated.
    carrier_freq_ghz : float
        Carrier frequency in GHz (default 77.0 for SENTINEL Mesh).
    polarization : str
        'HH' or 'VV'.  Only 'HH' is currently parameterised.
    prf_hz : int
        Pulse repetition frequency in Hz.
    cpi_pulses : int
        Number of pulses per coherent processing interval.
    n_range_bins : int
        Number of range bins to simulate.
    cnr_db : float
        Clutter-to-noise ratio in dB.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        bistatic_angle_deg: float,
        carrier_freq_ghz: float = 77.0,
        polarization: str = "HH",
        prf_hz: int = 1000,
        cpi_pulses: int = 128,
        n_range_bins: int = 256,
        cnr_db: float = 15.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 60.0 <= bistatic_angle_deg <= 120.0:
            raise ValueError(
                f"bistatic_angle_deg must be in [60, 120], got {bistatic_angle_deg}"
            )
        if polarization.upper() != "HH":
            raise NotImplementedError("Only HH polarization is parameterised.")

        self.bistatic_angle_deg = float(bistatic_angle_deg)
        self.carrier_freq_ghz = float(carrier_freq_ghz)
        self.polarization = polarization.upper()
        self.prf_hz = int(prf_hz)
        self.cpi_pulses = int(cpi_pulses)
        self.n_range_bins = int(n_range_bins)
        self.cnr_db = float(cnr_db)
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # Pre-compute empirical parameters for this geometry.
        self._params = self._get_empirical_params()

    # ── empirical parameter interpolation ─────────────────────────────

    def _get_empirical_params(self) -> dict[str, float]:
        """Return interpolated & frequency-scaled empirical parameters.

        Linearly interpolates between the three NetRAD anchor points
        (60°, 90°, 120°) for ν, σ_PSD, and CoG.

        Returns
        -------
        dict with keys:
            nu            – K-distribution shape parameter (dimensionless)
            sigma_psd_hz  – PSD half-width at 77 GHz (Hz)
            cog_hz        – Centre-of-gravity Doppler at 77 GHz (Hz)
            B_gradient    – CoG-intensity gradient (Hz / texture unit).
                            Approximated as -cog_hz / 10.0 pending better
                            empirical data (see KNOWN_LIMITATIONS).
        """
        beta = self.bistatic_angle_deg
        angles = np.array(_ANCHOR_ANGLES, dtype=float)

        # Extract anchor values for each parameter.
        nu_vals = np.array([NETRAD_PARAMS_HH[a]["nu"] for a in _ANCHOR_ANGLES])
        sigma_vals = np.array(
            [NETRAD_PARAMS_HH[a]["sigma_psd_hz_sband"] for a in _ANCHOR_ANGLES]
        )
        cog_vals = np.array(
            [NETRAD_PARAMS_HH[a]["cog_hz_sband"] for a in _ANCHOR_ANGLES]
        )

        # Linear interpolation.
        nu = float(np.interp(beta, angles, nu_vals))
        sigma_psd_hz = float(np.interp(beta, angles, sigma_vals)) * FREQ_SCALE
        cog_hz = float(np.interp(beta, angles, cog_vals)) * FREQ_SCALE

        # B gradient approximation (see KNOWN_LIMITATIONS).
        b_gradient = -cog_hz / 10.0

        return {
            "nu": nu,
            "sigma_psd_hz": sigma_psd_hz,
            "cog_hz": cog_hz,
            "B_gradient": b_gradient,
        }

    # ── texture generation ────────────────────────────────────────────

    def generate_texture(self) -> np.ndarray:
        """Generate correlated texture envelope.

        Uses a moving-average approximation of an AR(1) Gamma process:
        1. Draw i.i.d. samples from Gamma(shape=ν, scale=1/ν) so that
           the marginal distribution has mean = 1.
        2. Smooth with a moving-average kernel of width
           ``int(τ_texture × PRF)`` to introduce temporal correlation
           matching the ~3 s texture decorrelation time.
        3. Re-normalise each range bin so that mean intensity = 1.

        **Approximation note:** A true AR(1) Gamma process is
        non-trivial.  The moving-average approach preserves the correct
        marginal mean and introduces realistic slowly-varying texture
        but does not exactly reproduce the Gamma marginal distribution.

        Returns
        -------
        np.ndarray, shape ``(n_range_bins, cpi_pulses)``
            Strictly positive texture intensity values with mean ≈ 1.
        """
        nu = self._params["nu"]
        tau_texture_s = 3.0  # texture decorrelation time (seconds)
        window = max(1, int(tau_texture_s * self.prf_hz))

        # Number of raw samples needed so that *valid* convolution
        # yields at least cpi_pulses outputs.
        n_gen = window + self.cpi_pulses - 1

        texture = np.empty((self.n_range_bins, self.cpi_pulses), dtype=float)

        for r in range(self.n_range_bins):
            # i.i.d. Gamma samples with mean = 1, variance = 1/ν.
            raw = self._rng.gamma(shape=nu, scale=1.0 / nu, size=n_gen)

            if window > 1:
                kernel = np.ones(window) / window
                smoothed = np.convolve(raw, kernel, mode="valid")
                # np.convolve valid: len = n_gen - window + 1 = cpi_pulses
                smoothed = smoothed[: self.cpi_pulses]
            else:
                smoothed = raw[: self.cpi_pulses]

            # Re-normalise to mean = 1.
            m = smoothed.mean()
            if m > 0.0:
                smoothed /= m

            texture[r, :] = smoothed

        return texture

    # ── speckle generation ────────────────────────────────────────────

    def generate_speckle(self) -> np.ndarray:
        """Generate i.i.d. complex Gaussian speckle.

        At 77 GHz the speckle decorrelation time (≈ 1.27 ms) is shorter
        than one PRI at typical drone PRFs (1–10 kHz), so speckle is
        modelled as independent between pulses.

        Each sample is drawn from CN(0, 1):
            w = (randn + j·randn) / √2

        so that E[|w|²] = 1  (unit mean power).

        Returns
        -------
        np.ndarray, shape ``(n_range_bins, cpi_pulses)``
            Complex array with unit mean power per cell.
        """
        shape = (self.n_range_bins, self.cpi_pulses)
        return (
            self._rng.standard_normal(shape)
            + 1j * self._rng.standard_normal(shape)
        ) / np.sqrt(2.0)

    # ── clutter voltage (compound K-distribution) ─────────────────────

    def generate_clutter_voltage(self) -> np.ndarray:
        """Combine texture and speckle into complex clutter voltage.

        The compound K-distribution model produces clutter as:

            c[r, t] = √(texture[r, t]) · speckle[r, t] · √(clutter_power)

        where ``clutter_power = noise_power × 10^(CNR_dB / 10)`` and
        ``noise_power = 1.0`` (normalised).

        Returns
        -------
        np.ndarray, shape ``(n_range_bins, cpi_pulses)``
            Complex clutter voltage array.
        """
        noise_power = 1.0
        clutter_power = noise_power * 10.0 ** (self.cnr_db / 10.0)

        texture = self.generate_texture()
        speckle = self.generate_speckle()

        return np.sqrt(texture) * speckle * np.sqrt(clutter_power)

    # ── Doppler PSD model ─────────────────────────────────────────────

    def generate_doppler_psd(
        self, texture_slice: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate the model Gaussian Doppler PSD for one range bin.

        This returns the *analytical model* PSD (not an FFT of voltage
        data).  The PSD follows:

            G(f) = (x̄ / √(2π σ²)) · exp(-(f - m_f)² / (2σ²))

        where ``m_f = CoG + B · x̄`` is the intensity-dependent mean
        Doppler (Centre of Gravity).

        Parameters
        ----------
        texture_slice : np.ndarray, shape ``(cpi_pulses,)``
            1-D texture intensity array for a single range bin across
            the CPI.

        Returns
        -------
        f_axis : np.ndarray, shape ``(cpi_pulses,)``
            Doppler frequency axis in Hz (centred via fftfreq).
        G : np.ndarray, shape ``(cpi_pulses,)``
            Model PSD values (linear power, not dB).
        """
        sigma = self._params["sigma_psd_hz"]
        cog = self._params["cog_hz"]
        B = self._params["B_gradient"]

        x_mean = float(np.mean(texture_slice))

        # Intensity-dependent CoG (linear relationship).
        m_f = cog + B * x_mean

        f_axis = np.fft.fftfreq(self.cpi_pulses, d=1.0 / self.prf_hz)

        G = (x_mean / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
            -((f_axis - m_f) ** 2) / (2.0 * sigma ** 2)
        )

        return f_axis, G

    # ── range-Doppler map ─────────────────────────────────────────────

    def compute_range_doppler_map(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a full range-Doppler power map.

        Pipeline:
        1. Generate compound-K clutter voltage ``(n_range, n_pulse)``.
        2. Add thermal noise: ``n = (randn + j·randn) / √2``.
        3. Apply a Hanning window along the pulse (Doppler) dimension.
        4. FFT along pulses → Doppler spectra per range bin.
        5. Compute power in dB: ``10 · log10(|FFT|²)``.

        Returns
        -------
        power_map_db : np.ndarray, shape ``(n_range_bins, cpi_pulses)``
            Power in dB (10 log10 of squared magnitude).
        doppler_axis : np.ndarray, shape ``(cpi_pulses,)``
            Doppler frequency axis in Hz.
        range_axis : np.ndarray, shape ``(n_range_bins,)``
            Range bin indices (0-based).
        """
        clutter = self.generate_clutter_voltage()

        # Additive thermal noise (unit power).
        noise = (
            self._rng.standard_normal(clutter.shape)
            + 1j * self._rng.standard_normal(clutter.shape)
        ) / np.sqrt(2.0)

        signal = clutter + noise

        # Hanning window along pulse dimension.
        window = np.hanning(self.cpi_pulses)
        signal *= window[np.newaxis, :]

        # FFT along pulse axis (axis=1 → Doppler).
        spectrum = np.fft.fft(signal, axis=1)

        power_linear = np.abs(spectrum) ** 2
        # Avoid log10(0) by clamping to a tiny floor.
        power_map_db = 10.0 * np.log10(np.maximum(power_linear, 1e-30))

        doppler_axis = np.fft.fftfreq(self.cpi_pulses, d=1.0 / self.prf_hz)
        range_axis = np.arange(self.n_range_bins, dtype=float)

        return power_map_db, doppler_axis, range_axis

    # ── shape-parameter estimator (MoM with CNR gating) ───────────────

    def estimate_shape_parameter(
        self,
        intensity: np.ndarray,
        noise_power: float = 1.0,
    ) -> float:
        """Estimate K-distribution shape parameter ν from observed data.

        Uses Method of Moments (MoM) with CNR-dependent gating to
        switch between estimators:

        * **CNR > 12 dB:** standard MoM
          ``ν = 1 / (MN2 − 1)``  where  ``MN2 = ⟨z²⟩ / ⟨z⟩²``
        * **3 dB ≤ CNR ≤ 12 dB:** K+Noise estimator
          ``ν = 2·(M1 − P_N)² / (M2 − 2·M1²)``
        * **CNR < 3 dB:** clutter is below noise floor → return ``np.inf``

        Parameters
        ----------
        intensity : np.ndarray
            Observed intensity (power) samples  ``|z|²``.
        noise_power : float
            Known / estimated thermal noise power level.

        Returns
        -------
        float
            Estimated shape parameter ν.  Returns ``np.inf`` when
            clutter is unresolvable (CNR < 3 dB).
        """
        intensity = np.asarray(intensity, dtype=float).ravel()
        M1 = float(np.mean(intensity))
        M2 = float(np.mean(intensity ** 2))

        cnr_linear = M1 / max(noise_power, 1e-30)
        cnr_db = 10.0 * np.log10(max(cnr_linear, 1e-30))

        if cnr_db < 3.0:
            # Noise-dominated — clutter shape is unresolvable.
            return float("inf")

        if cnr_db <= 12.0:
            # Transition region: K+Noise estimator.
            denom = M2 - 2.0 * M1 ** 2
            if abs(denom) < 1e-30:
                return float("inf")
            nu_est = 2.0 * (M1 - noise_power) ** 2 / denom
            return max(nu_est, 1e-6)

        # High-CNR regime: standard MoM for K-distributed intensity.
        # For I = texture × speckle  (Gamma(ν,1/ν) × Exp(1)):
        #   E[I]   = 1
        #   E[I²]  = E[x²]·E[|w|⁴] = (1+1/ν)·2 = 2 + 2/ν
        #   MN2    = E[I²]/E[I]² = 2 + 2/ν
        #   ⇒ ν    = 2 / (MN2 − 2)
        MN2 = M2 / max(M1 ** 2, 1e-30)
        denom = MN2 - 2.0
        if denom <= 0.0:
            return float("inf")
        return 2.0 / denom

    # ── diagnostics plot ──────────────────────────────────────────────

    def plot_diagnostics(self) -> None:
        """Generate a 2×2 diagnostic figure.

        Panels:
        - Top-left:     Range-Doppler power map (dB colormap)
        - Top-right:    Clutter amplitude histogram vs fitted K-dist PDF
        - Bottom-left:  Single range-bin PSD (FFT) vs Gaussian model PSD
        - Bottom-right: Texture autocorrelation (measured vs exponential)

        Requires ``matplotlib``.
        """
        import matplotlib.pyplot as plt

        nu = self._params["nu"]

        # --- generate data ---
        texture = self.generate_texture()
        speckle = self.generate_speckle()
        noise_power = 1.0
        clutter_power = noise_power * 10.0 ** (self.cnr_db / 10.0)
        clutter_v = np.sqrt(texture) * speckle * np.sqrt(clutter_power)

        noise = (
            self._rng.standard_normal(clutter_v.shape)
            + 1j * self._rng.standard_normal(clutter_v.shape)
        ) / np.sqrt(2.0)
        signal = clutter_v + noise

        window = np.hanning(self.cpi_pulses)
        spectrum = np.fft.fft(signal * window[np.newaxis, :], axis=1)
        power_lin = np.abs(spectrum) ** 2
        power_db = 10.0 * np.log10(np.maximum(power_lin, 1e-30))
        doppler_axis = np.fft.fftfreq(self.cpi_pulses, d=1.0 / self.prf_hz)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Clutter Diagnostics — β={self.bistatic_angle_deg:.0f}°, "
            f"ν={nu:.2f}, CNR={self.cnr_db:.0f} dB",
            fontsize=14, fontweight="bold",
        )

        # ---- Top-left: Range-Doppler map ----
        ax = axes[0, 0]
        d_sorted = np.argsort(doppler_axis)
        im = ax.pcolormesh(
            doppler_axis[d_sorted],
            np.arange(self.n_range_bins),
            power_db[:, d_sorted],
            shading="auto", cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="Power (dB)")
        ax.set_xlabel("Doppler (Hz)")
        ax.set_ylabel("Range bin")
        ax.set_title("Range-Doppler Map")

        # ---- Top-right: amplitude histogram vs K-distribution PDF ----
        ax = axes[0, 1]
        amplitudes = np.abs(clutter_v).ravel() / np.sqrt(clutter_power)
        ax.hist(amplitudes, bins=120, density=True, alpha=0.6,
                color="steelblue", label="Simulated")

        a_max = float(np.percentile(amplitudes, 99.5))
        a_grid = np.linspace(1e-6, a_max, 500)
        # K-dist amplitude PDF: p(a) = (4·ν^((ν+1)/2)/Γ(ν)) · a^ν · K_{ν-1}(2√ν·a)
        log_pdf = (
            np.log(4.0)
            + ((nu + 1.0) / 2.0) * np.log(nu)
            - special.gammaln(nu)
            + nu * np.log(a_grid)
            + np.log(np.maximum(special.kv(nu - 1.0, 2.0 * np.sqrt(nu) * a_grid), 1e-300))
        )
        pdf_vals = np.exp(log_pdf)
        pdf_vals = np.where(np.isfinite(pdf_vals), pdf_vals, 0.0)
        ax.plot(a_grid, pdf_vals, "r-", lw=2, label=f"K-dist PDF (ν={nu:.2f})")
        ax.set_xlabel("Normalised amplitude")
        ax.set_ylabel("Density")
        ax.set_title("Amplitude Histogram vs K-Distribution")
        ax.legend()
        ax.set_xlim(0, a_max)

        # ---- Bottom-left: single range-bin PSD vs Gaussian model ----
        ax = axes[1, 0]
        ref_bin = 0
        psd_fft = power_db[ref_bin, :]
        f_model, G_model = self.generate_doppler_psd(texture[ref_bin, :])
        G_model_db = 10.0 * np.log10(np.maximum(G_model, 1e-30))

        sort_idx = np.argsort(doppler_axis)
        ax.plot(doppler_axis[sort_idx], psd_fft[sort_idx],
                alpha=0.7, label="FFT PSD (bin 0)")
        sort_m = np.argsort(f_model)
        ax.plot(f_model[sort_m], G_model_db[sort_m], "r--", lw=2,
                label="Gaussian model")
        ax.set_xlabel("Doppler (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title("PSD: FFT vs Gaussian Model")
        ax.legend()

        # ---- Bottom-right: texture autocorrelation ----
        ax = axes[1, 1]
        tex_row = texture[ref_bin, :]
        tex_row = tex_row - tex_row.mean()
        acf_measured = np.correlate(tex_row, tex_row, mode="full")
        acf_measured = acf_measured[len(tex_row) - 1 :]
        if acf_measured[0] > 0:
            acf_measured /= acf_measured[0]

        lags = np.arange(len(acf_measured)) / self.prf_hz
        tau_texture = 3.0
        acf_model = np.exp(-lags / tau_texture)

        ax.plot(lags * 1e3, acf_measured, label="Measured ACF")
        ax.plot(lags * 1e3, acf_model, "r--", lw=2,
                label=f"exp(−τ/{tau_texture:.0f}s)")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Texture Autocorrelation")
        ax.legend()

        plt.tight_layout()
        plt.savefig("clutter_diagnostics.png", dpi=150)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Standalone: embed_target_in_clutter
# ═══════════════════════════════════════════════════════════════════════════

# Sea-surface reflection coefficient (accounts for phase inversion and
# slight scattering loss at 77 GHz over the ocean surface).
_GAMMA_SEA: float = -0.9


def embed_target_in_clutter(
    clutter_map: np.ndarray,
    target_range_bin: int,
    target_doppler_hz: float,
    target_rcs_dbsm: float,
    noise_power: float,
    prf_hz: float,
    carrier_freq_ghz: float = 77.0,
    target_absolute_range_m: Optional[float] = None,
    drone_alt_m: Optional[float] = None,
    target_alt_m: Optional[float] = None,
) -> np.ndarray:
    """Inject a coherent point-target signal into a clutter voltage array.

    The target is modelled as a complex exponential at the specified
    Doppler frequency, with amplitude derived from its RCS.  When the
    optional multipath parameters are provided, the signal is further
    modulated by a complex propagation factor *F* that captures the
    constructive / destructive interference between the direct path and
    the sea-surface-reflected path.

    **Multipath physics (Generalized Target model)**

    For a radar at altitude *h_r* and a target at altitude *h_t*
    separated by ground range *R_g*:

    1. Path-length difference:  ``ΔR ≈ 2·h_r·h_t / R_g``
    2. Phase difference:        ``ΔΦ = 2π·ΔR / λ``
    3. Propagation factor:      ``F = 1 + Γ·exp(−jΔΦ)``

    where ``Γ ≈ −0.9`` is the sea-surface reflection coefficient
    (phase-inverting with slight scattering loss).

    The effective target voltage is multiplied by *F*, producing
    large fading nulls when the two paths destructively interfere
    and ≈ 6 dB enhancement when they constructively add.

    Parameters
    ----------
    clutter_map : np.ndarray, shape ``(n_range, n_pulse)``
        Complex voltage array (modified in-place and returned).
    target_range_bin : int
        Range bin index where the target is located.
    target_doppler_hz : float
        Target radial-velocity Doppler shift in Hz.  May be aliased
        by the PRF — the complex exponential handles this naturally.
    target_rcs_dbsm : float
        Target radar cross-section in dBsm.
        Typical values:
        * Sea-skimming missile (Mach 0.8–0.9): −20 to −10 dBsm
        * Subsonic drone: −30 to −20 dBsm
        * Frigate-class vessel: +30 to +40 dBsm
    noise_power : float
        Normalised thermal noise power (used to scale target amplitude
        relative to the noise floor).
    prf_hz : float
        Pulse repetition frequency in Hz.
    carrier_freq_ghz : float
        Carrier frequency in GHz (default 77.0).
    target_absolute_range_m : float or None
        Slant range from drone to target in metres.  Required for
        multipath calculation (along with *drone_alt_m* and
        *target_alt_m*).  When ``None`` the multipath factor is
        not applied (legacy single-point behaviour).
    drone_alt_m : float or None
        Altitude of the radar drone in metres above sea level.
    target_alt_m : float or None
        Altitude of the target in metres above sea level.

    Returns
    -------
    np.ndarray
        The modified complex voltage array (same object as input).
    """
    n_pulses = clutter_map.shape[1]

    # Linear RCS from dBsm.
    rcs_linear = 10.0 ** (target_rcs_dbsm / 10.0)

    # Target amplitude (voltage) scaled relative to noise power.
    amplitude = np.sqrt(rcs_linear * noise_power)

    # Pulse index vector.
    pulse_idx = np.arange(n_pulses, dtype=float)

    # ── Multipath propagation factor ──────────────────────────────
    # Compute complex factor F only when all three geometry
    # parameters are supplied; otherwise fall back to F = 1 (no
    # multipath), preserving backward compatibility.
    F: complex = 1.0 + 0j

    if (
        target_absolute_range_m is not None
        and drone_alt_m is not None
        and target_alt_m is not None
    ):
        # Radar wavelength.
        lambda_m = 3.0e8 / (carrier_freq_ghz * 1.0e9)

        # Ground range via Pythagorean theorem (clamped to ≥ 1 m to
        # avoid division-by-zero when the target is directly below).
        delta_alt = drone_alt_m - target_alt_m
        R_g = np.sqrt(max(target_absolute_range_m**2 - delta_alt**2, 1.0))

        # Path-length difference between direct and reflected paths.
        delta_R = 2.0 * drone_alt_m * target_alt_m / R_g

        # Phase difference.
        delta_phi = 2.0 * np.pi * delta_R / lambda_m

        # Complex propagation factor.
        F = 1.0 + _GAMMA_SEA * np.exp(-1j * delta_phi)

    # Coherent complex-exponential target signal with multipath.
    target_signal = F * amplitude * np.exp(
        2j * np.pi * target_doppler_hz * pulse_idx / prf_hz
    )

    clutter_map[target_range_bin, :] += target_signal

    return clutter_map


# ═══════════════════════════════════════════════════════════════════════════
# CFARDetector — Cell-Averaging CFAR
# ═══════════════════════════════════════════════════════════════════════════

class CFARDetector:
    """Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector.

    Operates along the Doppler dimension of a range-Doppler power map.
    For each cell-under-test (CUT) the detector:

    1. Excludes ``guard_cells`` on each side of the CUT.
    2. Averages the power of ``training_cells`` on each side (outside
       the guard band).
    3. Computes an adaptive threshold:
       ``threshold = noise_estimate × α``
       where ``α = N · (P_fa^{-1/N} − 1)`` and ``N = 2 × training_cells``.
    4. Flags the CUT as a detection if its power exceeds the threshold.

    Parameters
    ----------
    guard_cells : int
        Number of guard cells on *each* side of the CUT (default 2).
    training_cells : int
        Number of training (reference) cells on *each* side of the
        CUT, outside the guard band (default 8).
    pfa : float
        Desired probability of false alarm (default 1e-4).
    """

    def __init__(
        self,
        guard_cells: int = 2,
        training_cells: int = 8,
        pfa: float = 1e-4,
    ) -> None:
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa

        # Total reference cells used in the CA-CFAR formula.
        N = 2 * training_cells
        self.threshold_factor: float = float(N * (pfa ** (-1.0 / N) - 1.0))

    def detect(self, power_map: np.ndarray) -> np.ndarray:
        """Run CA-CFAR detection on a range-Doppler power map.

        Detection is performed independently for each range bin along
        the Doppler dimension.  Boundary cells that lack a full set of
        training cells are *not* flagged (conservative approach).

        Parameters
        ----------
        power_map : np.ndarray, shape ``(n_range, n_doppler)``
            Power values in **linear** scale (not dB).

        Returns
        -------
        np.ndarray, shape ``(n_range, n_doppler)``
            Binary detection map (1 = detection, 0 = no detection),
            dtype ``np.int32``.
        """
        power_map = np.asarray(power_map, dtype=float)
        n_range, n_doppler = power_map.shape

        detections = np.zeros_like(power_map, dtype=np.int32)

        half_window = self.guard_cells + self.training_cells

        for r in range(n_range):
            row = power_map[r, :]
            for d in range(n_doppler):
                # Leading / trailing edges of the training window.
                left_start = d - half_window
                left_end = d - self.guard_cells
                right_start = d + self.guard_cells + 1
                right_end = d + half_window + 1

                # Skip CUTs without a full training window.
                if left_start < 0 or right_end > n_doppler:
                    continue

                # Collect training-cell power.
                train_left = row[left_start:left_end]
                train_right = row[right_start:right_end]
                noise_estimate = np.mean(
                    np.concatenate([train_left, train_right])
                )

                threshold = noise_estimate * self.threshold_factor

                if row[d] > threshold:
                    detections[r, d] = 1

        return detections


# ═══════════════════════════════════════════════════════════════════════════
# Training-data batch generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_batch(
    n_samples: int = 100,
    bistatic_angles: list[float] | None = None,
    include_target: bool = True,
    target_rcs_range_dbsm: tuple[float, float] = (-20.0, 15.0),
    output_dir: str = "./training_data",
    prf_hz: int = 1000,
    cpi_pulses: int = 128,
    n_range_bins: int = 256,
    cnr_db: float = 15.0,
    save_format: str = "npz",
) -> list[dict]:
    """Generate a batch of labelled range-Doppler samples for ML training.

    Each sample contains a range-Doppler power map (dB), a binary CFAR
    detection map, and metadata describing the simulation parameters
    and (optionally) an embedded point target.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    bistatic_angles : list[float] or None
        Pool of bistatic angles to randomly draw from.
        Defaults to ``[60, 90, 120]``.
    include_target : bool
        If True, each sample has a 50/50 chance of containing a target.
    target_rcs_range_dbsm : tuple[float, float]
        (min, max) RCS in dBsm for uniform random target RCS.
    output_dir : str
        Directory to save output files.
    prf_hz, cpi_pulses, n_range_bins, cnr_db
        Simulation parameters forwarded to ``BistaticClutterModel``.
    save_format : str
        ``'npz'`` (default) to save per-sample ``.npz`` files.

    Returns
    -------
    list[dict]
        List of metadata dicts, one per sample.
    """
    from tqdm import trange

    if bistatic_angles is None:
        bistatic_angles = [60.0, 90.0, 120.0]

    rng = np.random.default_rng()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfar = CFARDetector(guard_cells=2, training_cells=8, pfa=1e-4)
    metadata_list: list[dict] = []
    speed_of_sound_mps = 340.0
    c_light = 3e8

    for i in trange(n_samples, desc="Generating training batch"):
        beta = float(rng.choice(bistatic_angles))
        sample_cnr = float(rng.uniform(cnr_db - 5.0, cnr_db + 5.0))

        model = BistaticClutterModel(
            bistatic_angle_deg=beta,
            carrier_freq_ghz=77.0,
            prf_hz=prf_hz,
            cpi_pulses=cpi_pulses,
            n_range_bins=n_range_bins,
            cnr_db=sample_cnr,
        )

        clutter_v = model.generate_clutter_voltage()
        noise_power = 1.0

        # Decide whether this sample contains a target.
        target_present = include_target and bool(rng.random() < 0.5)
        tgt_range_bin = -1
        tgt_doppler_hz = 0.0
        tgt_rcs_dbsm = 0.0

        if target_present:
            tgt_range_bin = int(rng.integers(10, n_range_bins - 10))
            mach = float(rng.uniform(0.5, 2.5))
            # Monostatic Doppler then bistatic correction cos(β/2).
            doppler_mono = 2.0 * mach * speed_of_sound_mps * 77e9 / c_light
            tgt_doppler_hz = float(
                doppler_mono * np.cos(np.radians(beta / 2.0))
            )
            tgt_rcs_dbsm = float(
                rng.uniform(*target_rcs_range_dbsm)
            )
            clutter_v = embed_target_in_clutter(
                clutter_map=clutter_v,
                target_range_bin=tgt_range_bin,
                target_doppler_hz=tgt_doppler_hz,
                target_rcs_dbsm=tgt_rcs_dbsm,
                noise_power=noise_power,
                prf_hz=prf_hz,
            )

        # Add thermal noise, window, FFT.
        noise = (
            rng.standard_normal(clutter_v.shape)
            + 1j * rng.standard_normal(clutter_v.shape)
        ) / np.sqrt(2.0)
        signal = clutter_v + noise
        win = np.hanning(cpi_pulses)
        spectrum = np.fft.fft(signal * win[np.newaxis, :], axis=1)
        power_lin = np.abs(spectrum) ** 2
        power_db = 10.0 * np.log10(np.maximum(power_lin, 1e-30))

        det_map = cfar.detect(power_lin)

        params = model._params
        meta = {
            "sample_id": i,
            "bistatic_angle_deg": beta,
            "cnr_db": sample_cnr,
            "target_present": int(target_present),
            "target_range_bin": tgt_range_bin,
            "target_doppler_hz": tgt_doppler_hz,
            "target_rcs_dbsm": tgt_rcs_dbsm,
            "nu_shape": params["nu"],
            "sigma_psd_hz": params["sigma_psd_hz"],
            "cog_hz": params["cog_hz"],
        }
        metadata_list.append(meta)

        sample_file = out_path / f"sample_{i:04d}.npz"
        np.savez_compressed(
            sample_file,
            power_map_db=power_db.astype(np.float32),
            detection_map=det_map.astype(np.int8),
            metadata=meta,
        )

    # Write metadata CSV alongside samples.
    import csv

    csv_path = out_path / "metadata.csv"
    if metadata_list:
        fieldnames = list(metadata_list[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_list)

    print(f"\nSaved {len(metadata_list)} samples to {out_path.resolve()}")
    print(f"Metadata CSV: {csv_path.resolve()}")
    return metadata_list


# ═══════════════════════════════════════════════════════════════════════════
# Usage example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick single-model diagnostic.
    model = BistaticClutterModel(bistatic_angle_deg=90, cnr_db=12.0, seed=42)
    power_map, doppler_axis, range_axis = model.compute_range_doppler_map()
    print(f"Range-Doppler map shape: {power_map.shape}")
    print(f"  Doppler axis: {doppler_axis.min():.1f} .. {doppler_axis.max():.1f} Hz")
    print(f"  Power range : {power_map.min():.1f} .. {power_map.max():.1f} dB")
    model.plot_diagnostics()

    # Small training batch.
    batch = generate_training_batch(
        n_samples=10, output_dir="./sentinel_training_data"
    )
    print(f"Generated {len(batch)} samples")

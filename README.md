# SENTINEL Mesh — Radar Training-Data Simulator

Proof-of-concept pipeline that generates labelled 77 GHz bistatic
range-Doppler radar data of sea-skimming missiles in sea clutter, for
training and evaluating detection models.

The pipeline has four stages:

```
config/sim_config.yaml
        │
        ▼
1. main.py ──────────────►  logs/cruise_run_*.txt      (missile trajectories)
        │
        ▼
2. integrate_simulations.py ─►  integrated_output/*.npz  (RD videos + labels)
   (uses clutter_model.py)
        │
        ├──► 3. evaluate_detection.py ──► evaluation/pd_vs_range.{csv,png}
        │
        └──► 4. radar_visualizer.py      (interactive tactical console)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Fly 20 randomized cruise-only missile runs (≈ seconds)
python main.py

# 2. Fuse each trajectory with simulated sea clutter into a
#    range-Doppler video with per-frame ground truth (≈ minutes)
python integrate_simulations.py --seed 42

# 3. Evaluate CA-CFAR detection performance (Pd vs range, Pfa)
python evaluate_detection.py

# 4. Play back a run in the visualizer
python radar_visualizer.py
```

Tests: `pytest test_clutter_model.py` (physics invariants of the
clutter model, CFAR, and range equation).

## What each stage does

**Stage 1 — flight simulation** ([main.py](main.py), [src/](src/)).
2-D point-mass missile (RK4, transonic drag rise, thrust-based speed
hold) with PID sea-skim altitude hold ([src/guidance/guidance.py](src/guidance/guidance.py)).
Each run randomizes cruise altitude (4.5–60 m) and speed (Mach 0.8–3.5).

**Stage 2 — radar integration** ([integrate_simulations.py](integrate_simulations.py),
[clutter_model.py](clutter_model.py)).
A radar drone hovers at a forward picket position; each trajectory is
converted to drone-relative range/radial-velocity, resampled onto
128-pulse CPIs, and embedded into compound K-distribution bistatic sea
clutter (NetRAD-parameterised, frequency-scaled to 77 GHz). Target
amplitude follows the radar range equation (R⁻⁴), with Swerling 3 RCS
fluctuation and rough-sea (Miller-Brown-Vegh) multipath. Output `.npz`
files contain the RD video `(frames × range × Doppler)`, per-frame
ground truth (range/Doppler bin, RCS, SNR, multipath factor), and the
full radar configuration including the generation seed.

**Stage 3 — detection evaluation** ([evaluate_detection.py](evaluate_detection.py)).
Runs a vectorized CA-CFAR over every frame and reports probability of
detection binned by range plus the measured false-alarm rate.

**Stage 4 — visualization** ([radar_visualizer.py](radar_visualizer.py)).
Pygame console that plays back the RD videos in two modes (toggle with
`TAB`):

- **TECHNICAL** — full engineering view: range-Doppler heatmap with a dB
  colour scale, live CFAR detections, the ground-truth target cell, the
  confirmed Kalman tracks, the picket geometry, and complete per-frame
  telemetry + detector stats.
- **PRESENTATION** — a clean, self-explaining briefing view: a
  plain-language target card, a detection-status banner, and the
  picket-fence engagement profile with a live "detecting here" marker.

Detection status is **truth-gated** — the banner reads *TARGET TRACKED*
only when CFAR actually fires within a few cells of the known target
location for that frame. Nothing is hard-coded.

Controls: `SPACE` play/pause · `←/→` step · `TAB` switch view ·
`[ / ]` change dataset · `D` detections · `T` tracks · `C` CFAR
algorithm · `ESC/Q` quit. Click the timeline to scrub.

[npz_converter.py](npz_converter.py) dumps `.npz` contents to
CSV/JSON/text.

## Physics notes & known limitations

The clutter model's empirical grounding, frequency-scaling assumptions,
and open fidelity gaps are documented in `KNOWN_LIMITATIONS` at the top
of [clutter_model.py](clutter_model.py). Headline items:

- NetRAD ν values are S-band, sea state 3, 4.9 m resolution — treated
  as optimistic upper bounds at 77 GHz.
- The bistatic angle is held constant per run; in reality it varies
  along the trajectory.
- Range migration within a CPI is not modelled (a Mach 3 target crosses
  ~40 range bins per CPI — real processing would show range walk).
- Clutter CNR is constant across the swath; a σ⁰-based range profile
  (TSC/GIT) is the next planned fidelity step.
- `RadarSystemParams` defaults (50 W, 38 dBi, NF 8 dB) are *assumed*
  PoC values for a drone-mounted node, not a real design.

## Repository layout

| Path | Contents |
|---|---|
| `config/sim_config.yaml` | Flight-simulation parameters |
| `src/` | Missile physics, guidance, environment stubs |
| `logs/` | Stage-1 trajectory text files (3 samples tracked) |
| `integrated_output/` | Stage-2 RD videos (2 samples tracked) |
| `sentinel_training_data/` | Standalone single-frame training samples |
| `evaluation/` | Stage-3 outputs (untracked) |
| `test_clutter_model.py` | Physics-invariant test suite |

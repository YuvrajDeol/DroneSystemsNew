"""
evaluate_detection.py — CFAR Detection Performance over Integrated RD Videos
============================================================================

Proof-of-concept evaluation: runs a CA-CFAR detector over every frame of
every RD-video ``.npz`` in *input_dir* and reports:

* **Pd vs. range** — probability of detection binned by true target
  range.  A frame counts as detected when at least one CFAR hit falls
  within ``±range_tol`` range bins and ``±doppler_tol`` Doppler bins of
  the ground-truth target cell.
* **Pfa** — false-alarm rate measured over all cells outside the
  target neighbourhood, compared against the CFAR design Pfa.

Outputs ``pd_vs_range.csv`` and ``pd_vs_range.png`` in *output_dir*.

Usage
-----
    python evaluate_detection.py                       # defaults
    python evaluate_detection.py --input_dir integrated_output
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from clutter_model import CFARDetector


def evaluate_file(
    npz_path: Path,
    cfar: CFARDetector,
    range_tol: int = 2,
    doppler_tol: int = 2,
) -> tuple[list[tuple[float, int]], int, int]:
    """Evaluate one RD video.

    Returns
    -------
    detections : list of (range_m, detected) per target-in-bounds frame
    false_alarms : total CFAR hits outside the target neighbourhood
    tested_cells : total cells tested for false alarms
    """
    data = np.load(npz_path, allow_pickle=True)
    rd_video = data["rd_video"]          # (frames, range, doppler), dB
    metadata = list(data["metadata"])

    results: list[tuple[float, int]] = []
    false_alarms = 0
    tested_cells = 0

    for meta in metadata:
        f_idx = int(meta["frame"])
        power_db = rd_video[f_idx].astype(np.float64)
        power_lin = 10.0 ** (power_db / 10.0)

        det_map = cfar.detect_vectorized(power_lin)

        if meta["target_in_bounds"]:
            r_bin = int(meta["range_bin"])
            d_bin = int(meta["doppler_bin"])
            # rd_video is fftshifted along Doppler; doppler_bin is the
            # pre-shift index, so convert to the shifted index.
            n_dop = det_map.shape[1]
            d_bin_shifted = (d_bin + n_dop // 2) % n_dop

            r_lo = max(0, r_bin - range_tol)
            r_hi = min(det_map.shape[0], r_bin + range_tol + 1)
            d_idx = [
                (d_bin_shifted + k) % n_dop
                for k in range(-doppler_tol, doppler_tol + 1)
            ]
            window = det_map[r_lo:r_hi, :][:, d_idx]
            detected = int(window.any())
            results.append((float(meta["range_m"]), detected))

            # Mask the target neighbourhood out of the Pfa count.
            mask = np.ones_like(det_map, dtype=bool)
            mask[r_lo:r_hi, :] = False
            false_alarms += int(det_map[mask].sum())
            tested_cells += int(mask.sum())
        else:
            false_alarms += int(det_map.sum())
            tested_cells += det_map.size

    data.close()
    return results, false_alarms, tested_cells


def main() -> int:
    p = argparse.ArgumentParser(
        description="CFAR Pd/Pfa evaluation over integrated RD videos."
    )
    p.add_argument("--input_dir", type=str, default="integrated_output")
    p.add_argument("--output_dir", type=str, default="evaluation")
    p.add_argument("--pfa", type=float, default=1e-4,
                   help="CFAR design probability of false alarm.")
    p.add_argument("--guard", type=int, default=2)
    p.add_argument("--training", type=int, default=8)
    p.add_argument("--range_bin_km", type=float, default=0.5,
                   help="Width of Pd-vs-range histogram bins in km.")
    args = p.parse_args()

    in_path = Path(args.input_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_path.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {in_path.resolve()}")
        return 1

    cfar = CFARDetector(
        guard_cells=args.guard, training_cells=args.training, pfa=args.pfa
    )

    all_results: list[tuple[float, int]] = []
    total_fa = 0
    total_cells = 0

    for f in tqdm(npz_files, desc="Evaluating"):
        results, fa, cells = evaluate_file(f, cfar)
        all_results.extend(results)
        total_fa += fa
        total_cells += cells

    if not all_results:
        print("No target-in-bounds frames found — nothing to evaluate.")
        return 1

    ranges = np.array([r for r, _ in all_results])
    hits = np.array([d for _, d in all_results])

    bin_w = args.range_bin_km * 1000.0
    edges = np.arange(0.0, ranges.max() + bin_w, bin_w)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ranges >= lo) & (ranges < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append({
            "range_low_m": lo,
            "range_high_m": hi,
            "n_frames": n,
            "pd": float(hits[sel].mean()),
        })

    measured_pfa = total_fa / max(total_cells, 1)

    csv_path = out_path / "pd_vs_range.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── plot ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = [(r["range_low_m"] + r["range_high_m"]) / 2e3 for r in rows]
    pds = [r["pd"] for r in rows]
    counts = [r["n_frames"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(centers, pds, "o-", lw=2, color="steelblue")
    ax.set_xlabel("Target range (km)")
    ax.set_ylabel("Probability of detection")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"CA-CFAR Pd vs range — {len(npz_files)} runs, "
        f"{len(all_results)} frames\n"
        f"design Pfa = {args.pfa:.0e}, measured Pfa = {measured_pfa:.2e}"
    )
    for x, y, n in zip(centers, pds, counts):
        ax.annotate(str(n), (x, y), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, ha="center", alpha=0.6)
    fig.tight_layout()
    png_path = out_path / "pd_vs_range.png"
    fig.savefig(png_path, dpi=150)

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  DETECTION EVALUATION — {len(npz_files)} runs, "
          f"{len(all_results)} target frames")
    print(f"{'═'*60}")
    print(f"  Overall Pd          : {hits.mean():.3f}")
    print(f"  Design Pfa          : {args.pfa:.2e}")
    print(f"  Measured Pfa        : {measured_pfa:.2e}  "
          f"({total_fa} alarms / {total_cells} cells)")
    print(f"  Pd vs range         : {csv_path}")
    print(f"  Plot                : {png_path}")
    for r in rows:
        bar = "█" * int(r["pd"] * 30)
        print(f"   {r['range_low_m']/1e3:4.1f}–{r['range_high_m']/1e3:4.1f} km "
              f" Pd={r['pd']:.2f} {bar}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

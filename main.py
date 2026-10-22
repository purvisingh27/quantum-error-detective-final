"""Quantum Error Detective - run every experiment and write all output files.

Usage:
    python main.py                       # defaults: 8192 shots, outputs/ folder
    python main.py --shots 4096 --seed 7 --outdir results
"""

import argparse
from pathlib import Path

import pandas as pd

import experiments as ex
import plotting


def parse_args():
    p = argparse.ArgumentParser(description="Quantum Error Detective")
    p.add_argument("--shots", type=int, default=8192, help="shots per circuit run (default 8192)")
    p.add_argument("--seed", type=int, default=42, help="base random seed (default 42)")
    p.add_argument("--outdir", default="outputs", help="output folder (default: outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Circuit metrics ...")
    ex.save_json(ex.collect_circuit_metrics(args.seed), out / "circuit_metrics.json")

    print("[2/4] Noise sweep (gate / readout / combined) ...")
    sweep = ex.run_noise_sweep(args.shots, args.seed)
    sweep.to_csv(out / "noise_sweep.csv", index=False)

    print("[3/4] Readout-error mitigation study ...")
    mit, cal = ex.run_mitigation_study(args.shots, args.seed)
    mit.to_csv(out / "mitigation_summary.csv", index=False)
    cal.to_csv(out / "readout_calibration_matrix.csv", index=False)

    print("[4/4] Plots and metadata ...")
    plotting.plot_noise_vs_success(sweep, out / "noise_vs_success.png")
    plotting.plot_mitigation_comparison(mit, out / "mitigation_comparison.png")
    plotting.plot_noise_components(sweep, ex.COMPONENT_LEVEL, out / "noise_component_comparison.png")
    ex.save_json(ex.build_metadata(args.shots, args.seed), out / "run_metadata.json")

    pd.set_option("display.width", 140)
    print("\n=== Mitigation summary (success probability) ===")
    cols = ["circuit", "scenario", "ideal_success", "raw_success", "mitigated_success", "improvement"]
    print(mit[cols].round(3).to_string(index=False))
    print(f"\nAll files written to: {out.resolve()}")


if __name__ == "__main__":
    main()

"""Experiment runners for Quantum Error Detective.

Three experiments:
  1. noise sweep        - success probability vs noise strength, for gate-only,
                          readout-only and combined noise
  2. mitigation study   - raw vs readout-mitigated results at a fixed noise level
  3. circuit metrics    - depth and gate counts, before and after transpiling

"Noise level" L is one knob that maps onto the physical error rates:
    gate noise     : 1-qubit depolarizing error = L * GATE_SCALE (CX error = 10x that)
    readout noise  : P(read 1 | state 0) = L   (P(read 0 | state 1) = 1.5 * L)
    combined noise : both of the above at the same L
"""

import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from circuits import all_circuits, with_measurements
from mitigation import (
    build_calibration_matrix,
    counts_to_vector,
    mitigate_probabilities,
    success_from_vector,
)
from noise import BASIS_GATES, READOUT_ASYMMETRY, TWO_QUBIT_FACTOR, build_noise_model, transpile_for_noise

NOISE_TYPES = ["gate", "readout", "combined"]
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
GATE_SCALE = 0.2            # 1-qubit gate error = level * 0.2  (so CX error = level * 2)
COMPONENT_LEVEL = 0.06      # level used for the gate/readout/combined bar chart
MITIGATION_LEVEL = 0.08     # level used for the mitigation study
MITIGATION_SCENARIOS = ["readout", "combined"]
BOOTSTRAP_RESAMPLES = 300


# ---------------------------------------------------------------- helpers
def noise_params(noise_type: str, level: float):
    gate = level * GATE_SCALE if noise_type in ("gate", "combined") else 0.0
    readout = level if noise_type in ("readout", "combined") else 0.0
    return gate, readout


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def ideal_probabilities(spec) -> np.ndarray:
    """Exact noiseless output distribution (index = int(bitstring, 2))."""
    return np.asarray(Statevector(spec.circuit).probabilities())


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions (0 = identical)."""
    return float(0.5 * np.abs(p - q).sum())


def make_backend(gate_error: float, readout_error: float, seed: int) -> AerSimulator:
    return AerSimulator(noise_model=build_noise_model(gate_error, readout_error), seed_simulator=seed)


def run_counts(spec, backend, shots: int, seed: int) -> dict:
    tc = transpile_for_noise(with_measurements(spec.circuit), backend, seed)
    return backend.run(tc, shots=shots, seed_simulator=seed).result().get_counts()


def count_successes(counts: dict, expected) -> int:
    return int(sum(counts.get(b, 0) for b in expected))


# ---------------------------------------------------------------- experiment 1
def run_noise_sweep(shots: int, seed: int) -> pd.DataFrame:
    rows = []
    run_id = 0
    for spec in all_circuits():
        ideal = ideal_probabilities(spec)
        ideal_success = success_from_vector(ideal, spec.expected, spec.n_qubits)
        for noise_type in NOISE_TYPES:
            for level in NOISE_LEVELS:
                gate_err, ro_err = noise_params(noise_type, level)
                run_id += 1
                run_seed = seed + run_id
                backend = make_backend(gate_err, ro_err, run_seed)
                counts = run_counts(spec, backend, shots, run_seed)
                k = count_successes(counts, spec.expected)
                lo, hi = wilson_ci(k, shots)
                rows.append({
                    "circuit": spec.name,
                    "noise_type": noise_type,
                    "noise_level": level,
                    "gate_error_1q": gate_err,
                    "readout_error_p01": ro_err,
                    "shots": shots,
                    "ideal_success": ideal_success,
                    "success_prob": k / shots,
                    "ci_low": lo,
                    "ci_high": hi,
                    "tvd_from_ideal": tvd(counts_to_vector(counts, spec.n_qubits), ideal),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment 2
def run_mitigation_study(shots: int, seed: int):
    """Returns (summary DataFrame, calibration-matrix DataFrame in long format)."""
    rng = np.random.default_rng(seed)
    rows, cal_rows = [], []
    run_id = 1000
    for spec in all_circuits():
        n = spec.n_qubits
        ideal = ideal_probabilities(spec)
        ideal_success = success_from_vector(ideal, spec.expected, n)
        for scenario in MITIGATION_SCENARIOS:
            gate_err, ro_err = noise_params(scenario, MITIGATION_LEVEL)
            run_id += 1
            run_seed = seed + run_id
            backend = make_backend(gate_err, ro_err, run_seed)

            counts = run_counts(spec, backend, shots, run_seed)
            p_meas = counts_to_vector(counts, n)

            # Calibration is measured on the SAME noisy backend, so any gate noise in
            # the X preparation gates leaks into the matrix, as it would on hardware.
            A = build_calibration_matrix(backend, n, shots=shots, seed=run_seed)
            p_mit = mitigate_probabilities(p_meas, A)

            raw_k = count_successes(counts, spec.expected)
            raw_lo, raw_hi = wilson_ci(raw_k, shots)
            raw_s = raw_k / shots
            mit_s = success_from_vector(p_mit, spec.expected, n)

            # Bootstrap CI for the mitigated estimate (resample the measured counts;
            # calibration-matrix uncertainty is not included).
            boot = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                resampled = rng.multinomial(shots, p_meas) / shots
                boot.append(success_from_vector(mitigate_probabilities(resampled, A), spec.expected, n))
            mit_lo, mit_hi = np.percentile(boot, [2.5, 97.5])

            rows.append({
                "circuit": spec.name,
                "scenario": scenario,
                "noise_level": MITIGATION_LEVEL,
                "gate_error_1q": gate_err,
                "readout_error_p01": ro_err,
                "shots": shots,
                "ideal_success": ideal_success,
                "raw_success": raw_s,
                "raw_ci_low": raw_lo,
                "raw_ci_high": raw_hi,
                "mitigated_success": mit_s,
                "mit_ci_low": float(mit_lo),
                "mit_ci_high": float(mit_hi),
                "improvement": mit_s - raw_s,
                "raw_tvd": tvd(p_meas, ideal),
                "mitigated_tvd": tvd(p_mit, ideal),
            })

            for s in range(2 ** n):
                for m in range(2 ** n):
                    cal_rows.append({
                        "circuit": spec.name,
                        "scenario": scenario,
                        "n_qubits": n,
                        "prepared": format(s, f"0{n}b"),
                        "measured": format(m, f"0{n}b"),
                        "probability": float(A[m, s]),
                    })
    return pd.DataFrame(rows), pd.DataFrame(cal_rows)


# ---------------------------------------------------------------- experiment 3
def collect_circuit_metrics(seed: int) -> dict:
    backend = AerSimulator()
    metrics = {}
    for spec in all_circuits():
        logical = spec.circuit
        physical = transpile_for_noise(with_measurements(logical), backend, seed)
        metrics[spec.name] = {
            "description": spec.description,
            "n_qubits": spec.n_qubits,
            "expected_outcomes": sorted(spec.expected),
            "logical": {
                "depth": int(logical.depth()),
                "size": int(logical.size()),
                "gate_counts": {k: int(v) for k, v in logical.count_ops().items()},
            },
            "transpiled": {
                "basis_gates": BASIS_GATES,
                "depth": int(physical.depth()),
                "size": int(physical.size()),
                "two_qubit_gates": int(physical.num_nonlocal_gates()),
                "gate_counts": {k: int(v) for k, v in physical.count_ops().items()},
            },
        }
    return metrics


def build_metadata(shots: int, seed: int) -> dict:
    return {
        "project": "Quantum Error Detective",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "simulator": "AerSimulator (noisy, shot-based)",
        "shots_per_circuit": shots,
        "base_seed": seed,
        "circuits": [s.name for s in all_circuits()],
        "noise_types": NOISE_TYPES,
        "noise_levels": NOISE_LEVELS,
        "noise_model": {
            "gate_error_1q_per_level": GATE_SCALE,
            "two_qubit_error_factor": TWO_QUBIT_FACTOR,
            "readout_asymmetry_p10_over_p01": READOUT_ASYMMETRY,
            "gate_error_type": "depolarizing on sx, x (1q) and cx (2q); rz is noiseless",
            "readout_error_type": "asymmetric bit-flip on every qubit",
        },
        "component_comparison_level": COMPONENT_LEVEL,
        "mitigation": {
            "method": "calibration-matrix inversion (pseudo-inverse, clip negatives, renormalise)",
            "level": MITIGATION_LEVEL,
            "scenarios": MITIGATION_SCENARIOS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_interval_note": "Raw: Wilson 95%. Mitigated: bootstrap 95% over measured counts "
                                        "(calibration uncertainty not included).",
        },
        "note": "All results come from simulation, not real quantum hardware.",
    }


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

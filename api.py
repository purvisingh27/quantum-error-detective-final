"""FastAPI backend for the Quantum Error Detective web dashboard.
All displayed run results are computed by the existing Qiskit Aer experiment code.
"""
from __future__ import annotations
import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit import transpile
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qiskit import QuantumCircuit

import experiments as ex
from circuits import CircuitSpec, all_circuits, with_measurements
from mitigation import build_calibration_matrix, calibration_circuits, counts_to_vector, mitigate_probabilities, success_from_vector
from noise import build_noise_model, transpile_for_noise

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "webapp" / "static"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
app = FastAPI(title="Quantum Error Detective API", version="1.0.0", description="Interactive Qiskit Aer quantum-noise experiments")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

class RunRequest(BaseModel):
    circuit: str = Field(default="Bell")
    noise_type: str = Field(default="combined")
    noise_level: float = Field(default=0.08, ge=0.0, le=0.2)
    shots: int = Field(default=4096, ge=256, le=32768)
    seed: int = Field(default=42, ge=0, le=2147483000)
    device_profile: str = Field(default="generic")
    custom_qasm: str | None = Field(default=None, max_length=20000)
    expected_outcomes: list[str] = Field(default_factory=list, max_length=256)

class ArenaRequest(BaseModel):
    circuit: str = "Bell"
    noise_type: str = "combined"
    noise_level: float = Field(default=0.08, ge=0.0, le=0.2)
    shots: int = Field(default=2048, ge=256, le=8192)
    seed: int = Field(default=42, ge=0, le=2147483000)

class SweepRequest(BaseModel):
    circuit: str = "Bell"
    noise_type: str = "combined"
    shots: int = Field(default=2048, ge=256, le=16384)
    seed: int = Field(default=42, ge=0, le=2147483000)
    device_profile: str = Field(default="generic")


def _get_spec(name: str):
    for spec in all_circuits():
        if spec.name.lower() == name.lower():
            return spec
    # Permit GHZ sizes 2–8 without adding a separate hard-coded option for every size.
    if name.lower().startswith("ghz"):
        try:
            n = int(name[3:])
        except ValueError:
            n = 0
        if 2 <= n <= 8:
            from circuits import ghz_circuit
            return ghz_circuit(n)
    raise HTTPException(status_code=400, detail="Unknown circuit. Choose Bell, GHZ3, GHZ5, GHZ8, or Grover2.")


def _custom_spec(qasm: str, expected_outcomes: list[str]):
    try:
        qc = QuantumCircuit.from_qasm_str(qasm)
        qc = qc.remove_final_measurements(inplace=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse OpenQASM 2.0 or remove final measurements: {exc}")
    if qc.num_qubits < 1 or qc.num_qubits > 8:
        raise HTTPException(status_code=400, detail="Custom circuits must contain between 1 and 8 qubits.")
    if qc.num_clbits:
        raise HTTPException(status_code=400, detail="Please use a circuit with only final measurements; mid-circuit/classical control is not supported in this prototype.")
    n = qc.num_qubits
    cleaned = sorted({str(x).strip().replace(" ", "") for x in expected_outcomes if str(x).strip()})
    if not cleaned:
        raise HTTPException(status_code=400, detail="Enter at least one expected output bitstring, comma-separated (for example: 00,11).")
    if any(len(x) != n or any(ch not in "01" for ch in x) for x in cleaned):
        raise HTTPException(status_code=400, detail=f"Expected bitstrings must each contain exactly {n} binary digits.")
    return CircuitSpec(name="Custom QASM", circuit=qc, expected=frozenset(cleaned), description=f"User-supplied OpenQASM circuit with {n} qubits; expected outputs: {', '.join(cleaned)}.")


def _make_backend_profile(device_profile: str, gate_error: float, readout_error: float, seed: int, n_qubits: int):
    """Return (simulator, optional transpilation target, profile label).

    IBM fake-provider backends use published/snapshotted backend properties and need
    qiskit-ibm-runtime installed. They are still simulations, not live hardware.
    """
    if device_profile == "ibm_fake_manila":
        if n_qubits > 5:
            raise HTTPException(status_code=400, detail="The FakeManila snapshot has 5 qubits. Choose GHZ3/GHZ5 or switch to Generic Aer for GHZ8.")
        try:
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            target = FakeManilaV2()
            simulator = AerSimulator.from_backend(target, seed_simulator=seed)
            return simulator, target, "IBM FakeManilaV2 noise snapshot (simulated; not live hardware)"
        except Exception as exc:
            raise HTTPException(status_code=400, detail="IBM fake-device mode needs qiskit-ibm-runtime installed. Run: python -m pip install qiskit-ibm-runtime. Details: " + str(exc))
    if device_profile != "generic":
        raise HTTPException(status_code=400, detail="device_profile must be generic or ibm_fake_manila")
    return ex.make_backend(gate_error, readout_error, seed), None, "Generic configurable Qiskit Aer noise model"


def _run_counts_profile(spec, backend, seed: int, transpile_target=None):
    measured = with_measurements(spec.circuit)
    if transpile_target is not None:
        compiled = transpile(measured, backend=transpile_target, optimization_level=1, seed_transpiler=seed)
    else:
        compiled = transpile_for_noise(measured, backend, seed)
    return backend.run(compiled, shots=2048, seed_simulator=seed).result().get_counts()


def _calibration_matrix_profile(backend, n: int, shots: int, seed: int, transpile_target=None, tensor: bool = False):
    """Sample full calibration for small circuits; tensor single-qubit calibration for larger GHZ states."""
    if not tensor:
        cals = calibration_circuits(n)
        compiled = [transpile(c, backend=transpile_target, optimization_level=1, seed_transpiler=seed + i)
                    for i, c in enumerate(cals)] if transpile_target is not None else [transpile_for_noise(c, backend, seed + i) for i, c in enumerate(cals)]
        result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
        A = np.zeros((2 ** n, 2 ** n))
        for state in range(2 ** n):
            A[:, state] = counts_to_vector(result.get_counts(state), n)
        return A
    matrices = []
    for q in range(n):
        cols = []
        for prepared in (0, 1):
            qc = __import__('qiskit').QuantumCircuit(n, n, name=f"tensor_cal_q{q}_{prepared}")
            if prepared:
                qc.x(q)
            qc.measure(range(n), range(n))
            if transpile_target is not None:
                compiled = transpile(qc, backend=transpile_target, optimization_level=1, seed_transpiler=seed + q * 2 + prepared)
            else:
                compiled = transpile_for_noise(qc, backend, seed + q * 2 + prepared)
            counts = backend.run(compiled, shots=shots, seed_simulator=seed + q * 2 + prepared).result().get_counts()
            vec = counts_to_vector(counts, n)
            # Keep only the measured bit for the calibrated qubit; other bits are marginalized.
            p1 = sum(count for bitstring, count in counts.items() if int(bitstring.replace(" ", ""), 2) & (1 << q)) / shots
            p0 = 1.0 - p1
            cols.append(np.array([p0, p1]))
        # Convert marginal outcome probabilities to a 2x2 matrix with measured rows, prepared columns.
        matrices.append(np.column_stack(cols))
    A = np.array([[1.0]])
    for q in reversed(range(n)):
        A = np.kron(A, matrices[q])
    return A


def _diagnose_noise(calibration, raw_tvd, mitigated_tvd):
    diag = np.diag(calibration)
    readout_error = float(np.clip(1.0 - np.mean(diag), 0.0, 1.0))
    residual = float(mitigated_tvd)
    if readout_error >= 0.035 and residual >= 0.08:
        label = "Likely combined readout + gate effects"
        why = "Calibration shows non-trivial measurement confusion, and substantial distribution error remains after readout correction."
    elif readout_error >= 0.035:
        label = "Readout-dominant pattern"
        why = "The calibration matrix shows measurement confusion; the corrected distribution is comparatively close to the ideal reference."
    elif residual >= 0.08:
        label = "Gate-dominant / unmodelled circuit error pattern"
        why = "Readout confusion is relatively small, but notable error remains after measurement correction. In this simulator, gate noise is a likely cause."
    else:
        label = "Low or inconclusive noise signature"
        why = "The current run does not clearly separate a dominant error source. Repeat with more shots or compare controlled noise scenarios."
    return {"label": label, "estimated_readout_error": round(readout_error, 4), "residual_tvd_after_readout_mitigation": round(residual, 4), "raw_tvd": round(float(raw_tvd), 4), "explanation": why, "caveat": "Heuristic diagnosis from simulated calibration and distributions; not a uniquely identifiable physical diagnosis."}


def _plain_explainer(result):
    change = result["mitigated_success"] - result["raw_success"]
    if change > 0.02:
        effect = f"Readout correction raised the estimated success rate by {change*100:.1f} percentage points in this sample."
    elif change < -0.02:
        effect = f"Readout correction lowered the estimated success rate by {abs(change)*100:.1f} percentage points in this sample, which can happen when calibration noise is amplified."
    else:
        effect = "Readout correction changed the estimated success rate only slightly in this sample."
    return effect + " This is a simulator experiment; compare confidence intervals and repeat runs before making a strong claim."


def _recommend(raw: float, mitigated: float, ideal: float, raw_tvd: float, mitigated_tvd: float, noise_type: str):
    delta = mitigated - raw
    tvd_delta = raw_tvd - mitigated_tvd
    if delta > 0.02 and tvd_delta > 0.005:
        message = "Mitigation improved the estimated success probability and moved the distribution closer to the ideal distribution in this run."
        status = "positive"
    elif delta < -0.02 or tvd_delta < -0.005:
        message = "Mitigation did not improve both measures in this run. Calibration sampling and inversion can add estimation error; compare repeated runs before drawing a conclusion."
        status = "caution"
    else:
        message = "The change is small in this run. Repeat with more shots and inspect the confidence intervals before deciding whether mitigation helps."
        status = "neutral"
    if noise_type == "gate":
        message += " This run uses gate noise only; readout mitigation cannot directly undo gate errors."
    elif noise_type == "combined":
        message += " Because gate and readout noise are both present, readout mitigation cannot recover all gate-induced errors."
    else:
        message += " This is a readout-noise-only simulation; it does not establish performance on real hardware."
    return {"status": status, "message": message, "success_change_percentage_points": round(delta * 100, 2), "tvd_reduction": round(tvd_delta, 4)}

@app.get("/", response_class=HTMLResponse)
def home():
    return (STATIC / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Quantum Error Detective", "execution": "Qiskit Aer simulator", "timestamp_utc": datetime.now(timezone.utc).isoformat()}

@app.get("/api/summary")
def summary():
    result: dict[str, Any] = {"execution": "Qiskit Aer simulator", "saved_results_available": False}
    for filename, key in [("mitigation_summary.csv", "mitigation"), ("noise_sweep.csv", "sweep")]:
        path = OUTPUTS / filename
        if path.exists():
            try:
                df = pd.read_csv(path).replace({np.nan: None})
                result[key] = df.to_dict(orient="records")
                result["saved_results_available"] = True
            except Exception:
                result[key] = []
        else:
            result[key] = []
    for filename, key in [("circuit_metrics.json", "metrics"), ("run_metadata.json", "metadata")]:
        path = OUTPUTS / filename
        if path.exists():
            try: result[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception: result[key] = {}
        else: result[key] = {}
    result["circuits"] = [{"name": s.name, "qubits": s.n_qubits, "description": s.description, "expected_outcomes": sorted(s.expected)} for s in all_circuits()]
    return result

@app.post("/api/run")
def run_experiment(req: RunRequest):
    if req.noise_type not in ex.NOISE_TYPES:
        raise HTTPException(status_code=400, detail="noise_type must be gate, readout, or combined")
    spec = _custom_spec(req.custom_qasm, req.expected_outcomes) if req.circuit.lower() == "custom" and req.custom_qasm else _get_spec(req.circuit)
    started = time.perf_counter()
    gate_err, ro_err = ex.noise_params(req.noise_type, req.noise_level)
    backend, target, profile_label = _make_backend_profile(req.device_profile, gate_err, ro_err, req.seed, spec.n_qubits)
    measured = with_measurements(spec.circuit)
    compiled = transpile(measured, backend=target, optimization_level=1, seed_transpiler=req.seed) if target is not None else transpile_for_noise(measured, backend, req.seed)
    counts = backend.run(compiled, shots=req.shots, seed_simulator=req.seed).result().get_counts()
    raw_vector = counts_to_vector(counts, spec.n_qubits)
    ideal_vector = ex.ideal_probabilities(spec)
    raw_successes = ex.count_successes(counts, spec.expected)
    raw_success = raw_successes / req.shots
    raw_low, raw_high = ex.wilson_ci(raw_successes, req.shots)
    raw_tvd = ex.tvd(raw_vector, ideal_vector)
    # Calibrate on the same simulator noise model, using actual sampled calibration counts.
    calibration = _calibration_matrix_profile(backend, spec.n_qubits, shots=req.shots, seed=req.seed + 10000, transpile_target=target, tensor=spec.n_qubits >= 4)
    calibration_inverse = np.linalg.pinv(calibration)
    mitigated_vector = mitigate_probabilities(raw_vector, calibration, inverse=calibration_inverse)
    mitigated_success = success_from_vector(mitigated_vector, spec.expected, spec.n_qubits)
    # Bootstrap measured counts to estimate uncertainty in the mitigated success estimate.
    rng = np.random.default_rng(req.seed + 20000)
    boot = []
    for _ in range(200):
        resampled = rng.multinomial(req.shots, raw_vector) / req.shots
        boot.append(success_from_vector(mitigate_probabilities(resampled, calibration, inverse=calibration_inverse), spec.expected, spec.n_qubits))
    mit_low, mit_high = np.percentile(boot, [2.5, 97.5])
    ideal_success = success_from_vector(ideal_vector, spec.expected, spec.n_qubits)
    mit_tvd = ex.tvd(mitigated_vector, ideal_vector)
    all_bits = [format(i, f"0{spec.n_qubits}b") for i in range(2**spec.n_qubits)]
    histogram = [{"bitstring": b, "raw_probability": float(raw_vector[int(b, 2)]), "mitigated_probability": float(mitigated_vector[int(b, 2)]), "ideal_probability": float(ideal_vector[int(b, 2)])} for b in all_bits]
    matrix = [[round(float(calibration[m, s]), 5) for m in range(2**spec.n_qubits)] for s in range(2**spec.n_qubits)]
    metrics = ex.collect_circuit_metrics(req.seed).get(spec.name, {})
    result = {
        "run_id": datetime.now(timezone.utc).strftime("QED-%Y%m%d-%H%M%S") + f"-{req.seed}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "circuit": spec.name, "description": spec.description, "n_qubits": spec.n_qubits,
        "noise_type": req.noise_type, "noise_level": req.noise_level, "device_profile": req.device_profile, "execution_profile": profile_label, "gate_error_1q": gate_err,
        "readout_error_p01": ro_err, "shots": req.shots, "seed": req.seed,
        "ideal_success": ideal_success, "raw_success": raw_success,
        "raw_ci_low": raw_low, "raw_ci_high": raw_high,
        "mitigated_success": mitigated_success, "mitigated_ci_low": float(mit_low), "mitigated_ci_high": float(mit_high),
        "improvement": mitigated_success - raw_success, "raw_tvd": raw_tvd, "mitigated_tvd": mit_tvd,
        "counts": {b: int(counts.get(b, 0)) for b in all_bits}, "histogram": histogram,
        "calibration_matrix": matrix, "calibration_labels": all_bits, "expected_outcomes": sorted(spec.expected),
        "metrics": metrics, "runtime_seconds": round(time.perf_counter() - started, 2),
        "recommendation": _recommend(raw_success, mitigated_success, ideal_success, raw_tvd, mit_tvd, req.noise_type),
        "diagnosis": _diagnose_noise(calibration, raw_tvd, mit_tvd),
        "plain_language_explainer": _plain_explainer({"raw_success": raw_success, "mitigated_success": mitigated_success}),
        "limitations": ["Simulation only; no live quantum hardware was used.", "Mitigated confidence interval bootstraps measured counts and does not include calibration-matrix uncertainty.", "Calibration matrix inversion may amplify sampling noise.", "Noise diagnosis is heuristic and is not a unique physical identification of the noise source."],
    }
    return result

@app.post("/api/mitigation-arena")
def mitigation_arena(req: ArenaRequest):
    """Compare raw, readout mitigation, digital noise-scaling ZNE, and combined estimates.

    ZNE here is an educational digital-noise-scaling approximation: gate-noise
    parameters are multiplied by 1, 2, and 3, then a linear fit is extrapolated
    to scale 0. This is not pulse-level gate folding and must be labelled as such.
    """
    if req.noise_type not in ex.NOISE_TYPES:
        raise HTTPException(status_code=400, detail="noise_type must be gate, readout, or combined")
    spec = _get_spec(req.circuit)
    ideal = ex.ideal_probabilities(spec)
    gate_err, ro_err = ex.noise_params(req.noise_type, req.noise_level)
    scales = [1.0, 2.0, 3.0]
    points = []
    for idx, scale in enumerate(scales):
        ge = min(gate_err * scale, 0.75)
        # Keep the readout error fixed while scaling gate noise.
        backend = ex.make_backend(ge, ro_err, req.seed + idx + 101)
        counts = ex.run_counts(spec, backend, req.shots, req.seed + idx + 101)
        raw_vec = counts_to_vector(counts, spec.n_qubits)
        raw_s = ex.count_successes(counts, spec.expected) / req.shots
        A = _calibration_matrix_profile(backend, spec.n_qubits, req.shots, req.seed + idx + 10001, tensor=spec.n_qubits >= 4)
        A_inverse = np.linalg.pinv(A)
        mit_vec = mitigate_probabilities(raw_vec, A, inverse=A_inverse)
        mit_s = success_from_vector(mit_vec, spec.expected, spec.n_qubits)
        points.append({"scale_factor": scale, "raw_success": float(raw_s), "readout_mitigated_success": float(mit_s), "raw_tvd": ex.tvd(raw_vec, ideal), "mitigated_tvd": ex.tvd(mit_vec, ideal)})
    x = np.array([p["scale_factor"] for p in points])
    raw_y = np.array([p["raw_success"] for p in points])
    mit_y = np.array([p["readout_mitigated_success"] for p in points])
    zne_raw = float(np.clip(np.polyfit(x, raw_y, 1)[1], 0, 1))
    zne_mit = float(np.clip(np.polyfit(x, mit_y, 1)[1], 0, 1))
    return {"circuit": spec.name, "noise_type": req.noise_type, "noise_level": req.noise_level, "shots": req.shots, "seed": req.seed,
        "ideal_success": success_from_vector(ideal, spec.expected, spec.n_qubits),
        "methods": [
            {"name": "Raw (no mitigation)", "success_estimate": points[0]["raw_success"], "tvd": points[0]["raw_tvd"], "interpretation": "Direct sampled result at the requested noise level."},
            {"name": "Readout mitigation", "success_estimate": points[0]["readout_mitigated_success"], "tvd": points[0]["mitigated_tvd"], "interpretation": "Corrects measurement bias using a sampled calibration matrix."},
            {"name": "ZNE (digital noise scaling)", "success_estimate": zne_raw, "tvd": None, "interpretation": "Linear extrapolation of raw success estimates at gate-noise scales 1, 2, and 3."},
            {"name": "Readout + ZNE", "success_estimate": zne_mit, "tvd": None, "interpretation": "Linear extrapolation of readout-mitigated estimates across the same gate-noise scales."}],
        "scale_points": points,
        "best_estimate_by_success": max([{"name": "Raw (no mitigation)", "success_estimate": points[0]["raw_success"]}, {"name": "Readout mitigation", "success_estimate": points[0]["readout_mitigated_success"]}, {"name": "ZNE (digital noise scaling)", "success_estimate": zne_raw}, {"name": "Readout + ZNE", "success_estimate": zne_mit}], key=lambda m: m["success_estimate"]),
        "note": "ZNE uses a simplified digital noise-parameter scaling model, not gate folding. Extrapolation can be biased or unstable; compare TVD where available, repeat seeds, and do not assume the highest success estimate is automatically most accurate."}


@app.post("/api/sweep")
def run_sweep(req: SweepRequest):
    if req.noise_type not in ex.NOISE_TYPES:
        raise HTTPException(status_code=400, detail="noise_type must be gate, readout, or combined")
    spec = _get_spec(req.circuit)
    rows = []
    for idx, level in enumerate(ex.NOISE_LEVELS):
        gate_err, ro_err = ex.noise_params(req.noise_type, float(level))
        backend, target, _profile_label = _make_backend_profile(req.device_profile, gate_err, ro_err, req.seed + idx + 1, spec.n_qubits)
        measured = with_measurements(spec.circuit)
        compiled = transpile(measured, backend=target, optimization_level=1, seed_transpiler=req.seed + idx + 1) if target is not None else transpile_for_noise(measured, backend, req.seed + idx + 1)
        counts = backend.run(compiled, shots=req.shots, seed_simulator=req.seed + idx + 1).result().get_counts()
        k = ex.count_successes(counts, spec.expected)
        low, high = ex.wilson_ci(k, req.shots)
        rows.append({"noise_level": float(level), "success_prob": k / req.shots, "ci_low": low, "ci_high": high, "shots": req.shots})
    return {"circuit": spec.name, "noise_type": req.noise_type, "shots": req.shots, "seed": req.seed, "points": rows, "note": "Every point is freshly sampled from Qiskit Aer using the selected noise model."}

@app.post("/api/export/csv")
def export_csv(payload: dict):
    result = payload.get("result")
    if not isinstance(result, dict):
        raise HTTPException(status_code=400, detail="Provide the experiment result in the request body.")
    flat = {k: v for k, v in result.items() if k not in ("histogram", "calibration_matrix", "counts", "metrics", "recommendation", "limitations")}
    flat["recommendation"] = result.get("recommendation", {}).get("message", "")
    flat["expected_outcomes"] = ";".join(result.get("expected_outcomes", []))
    flat["counts_json"] = json.dumps(result.get("counts", {}))
    flat["histogram_json"] = json.dumps(result.get("histogram", []))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(flat.keys()))
    writer.writeheader(); writer.writerow(flat)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=quantum-error-detective-{result.get('run_id','experiment')}.csv"})

@app.post("/api/export/report")
def export_report(payload: dict):
    r = payload.get("result")
    if not isinstance(r, dict):
        raise HTTPException(status_code=400, detail="Provide the experiment result in the request body.")
    rec = r.get("recommendation", {})
    lines = [
        "# Quantum Error Detective — Experiment Report", "",
        f"- **Run ID:** {r.get('run_id','N/A')}", f"- **Timestamp (UTC):** {r.get('timestamp_utc','N/A')}",
        f"- **Execution:** {r.get('execution_profile','Qiskit Aer simulator (simulated results; no live hardware)')}",
        f"- **Circuit:** {r.get('circuit','N/A')} ({r.get('n_qubits','?')} qubits)",
        f"- **Noise scenario:** {r.get('noise_type','N/A')} at level {r.get('noise_level','N/A')}",
        f"- **Shots:** {r.get('shots','N/A')} | **Seed:** {r.get('seed','N/A')}", "",
        "## Results", "", "| Metric | Value |", "|---|---:|",
        f"| Ideal success probability | {r.get('ideal_success',0):.4f} |",
        f"| Raw success probability | {r.get('raw_success',0):.4f} |",
        f"| Raw 95% Wilson CI | [{r.get('raw_ci_low',0):.4f}, {r.get('raw_ci_high',0):.4f}] |",
        f"| Mitigated success estimate | {r.get('mitigated_success',0):.4f} |",
        f"| Mitigated bootstrap 95% CI | [{r.get('mitigated_ci_low',0):.4f}, {r.get('mitigated_ci_high',0):.4f}] |",
        f"| Change after mitigation (percentage points) | {r.get('improvement',0)*100:.2f} |",
        f"| Raw total variation distance | {r.get('raw_tvd',0):.4f} |",
        f"| Mitigated total variation distance | {r.get('mitigated_tvd',0):.4f} |", "",
        "## Result-based recommendation", "", rec.get("message", "No recommendation available."), "",
        "## Noise Detective (heuristic)", "", json.dumps(r.get("diagnosis", {}), indent=2), "",
        "## Plain-language explainer", "", r.get("plain_language_explainer", "Not available."), "",
        "## Measurement counts", "", "| Bitstring | Count | Raw probability | Mitigated probability | Ideal probability |", "|---|---:|---:|---:|---:|"
    ]
    for row in r.get("histogram", []):
        lines.append(f"| {row['bitstring']} | {r.get('counts',{}).get(row['bitstring'],0)} | {row['raw_probability']:.4f} | {row['mitigated_probability']:.4f} | {row['ideal_probability']:.4f} |")
    lines += ["", "## Circuit metrics", "", "```json", json.dumps(r.get("metrics", {}), indent=2), "```", "", "## Limitations", ""]
    lines += [f"- {x}" for x in r.get("limitations", [])]
    lines += ["", "This report records one simulator experiment. Results may vary with shots and seed.", ""]
    filename = f"quantum-error-detective-{r.get('run_id','experiment')}.md"
    return Response(content="\n".join(lines), media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f"attachment; filename={filename}"})

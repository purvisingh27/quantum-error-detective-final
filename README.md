# Quantum Error Detective

A Python/Qiskit project that investigates **why quantum circuits return wrong answers** and
**whether readout-error mitigation can partly fix it**. Everything runs on a simulator
(Qiskit Aer) with *simulated* noise. No real quantum hardware is used.

## What it does

1. Builds three circuits: **Bell** (2 qubits), **GHZ** (3 qubits), **Grover search** (2 qubits, marked item `11`).
2. Computes the **ideal** (noiseless) result as a baseline.
3. Adds three kinds of simulated noise: **gate**, **readout**, and **combined**.
4. Measures **success probability** (with 95% confidence intervals) and total variation distance from ideal.
5. Applies **readout-error mitigation** using a calibration matrix and compares before/after.

## Quick start

```bash
pip install -r requirements.txt
python main.py                      # writes everything to outputs/
python main.py --shots 4096 --seed 7 --outdir results   # optional flags
```

Runtime is about 10 seconds.

## Files

| File | Role |
|---|---|
| `circuits.py` | Bell, GHZ, Grover circuits and their "correct" outcomes |
| `noise.py` | Gate (depolarizing) and readout (bit-flip) noise models |
| `mitigation.py` | Calibration matrix, inversion, projection to a valid distribution |
| `experiments.py` | Noise sweep, mitigation study, circuit metrics, metadata |
| `plotting.py` | The three figures |
| `main.py` | Runs everything |

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `noise_sweep.csv` | Success probability, CI and TVD vs noise level, per circuit and noise type |
| `noise_vs_success.png` | Success probability as noise grows |
| `mitigation_summary.csv` | Raw vs mitigated success, CIs, TVD, improvement |
| `mitigation_comparison.png` | Bar chart of raw vs mitigated |
| `noise_component_comparison.png` | Gate vs readout vs combined at one noise level |
| `readout_calibration_matrix.csv` | Estimated confusion matrices (long format) |
| `circuit_metrics.json` | Depth and gate counts, before and after transpiling |
| `run_metadata.json` | Library versions, seed, shots, all parameters |

## How the noise is defined

One **noise level L** drives all three noise types:

- **Gate noise:** depolarizing error of `0.2 × L` on 1-qubit gates (`sx`, `x`) and `2 × L` on `cx`. `rz` is noiseless.
- **Readout noise:** P(read 1 | state 0) = `L`, and P(read 0 | state 1) = `1.5 × L`.
- **Combined:** both at the same L.

These are illustrative values, not measurements of a specific device.

## How the mitigation works

1. Prepare each basis state `|s>`, measure it, and record how often each outcome `m` appears. This gives `A[m, s] = P(measure m | prepared s)`.
2. A real measured distribution is `p_measured = A · p_true`.
3. Estimate `p_true` with the pseudo-inverse of `A`, clip negatives to 0, and renormalise.

The calibration is run on the same noisy backend as the experiment. Gate noise on the X preparation gates therefore leaks into the matrix, as it would on real hardware.

## Results from a reference run (seed 42, 8192 shots, noise level 0.08)

| Circuit | Scenario | Raw | Mitigated |
|---|---|---|---|
| Bell | readout only | 0.81 | 0.99 |
| GHZ3 | readout only | 0.73 | 1.00 |
| Grover2 | readout only | 0.78 | 0.99 |
| Bell | gate + readout | 0.77 | 0.92 |
| GHZ3 | gate + readout | 0.62 | 0.83 |
| Grover2 | gate + readout | 0.60 | 0.76 |

Readout mitigation almost fully recovers the ideal result when only readout noise is present. With gate noise as well, it recovers part of the loss but not all of it, because it does not correct errors that happen inside the circuit. Re-run the project yourself and quote your own numbers.

## Limitations (be honest about these in your presentation)

- Simulation only. Real devices have correlated and drifting noise that this model does not capture.
- Mitigated confidence intervals come from bootstrapping the measured counts. They do not include uncertainty in the calibration matrix.
- Calibration-matrix inversion needs 2^n calibration circuits, so it does not scale to many qubits.
- This is an educational prototype, not a solution to quantum error correction.

## Ideas to extend it

- Run the same circuits on a real IBM Quantum backend and compare with the simulation.
- Add a different mitigation method such as zero-noise extrapolation.
- Add more circuits (QFT, larger GHZ) and sweep the number of qubits.


## Interactive web prototype (new)

The project now includes a separate frontend and FastAPI backend. The browser interface calls the actual Qiskit Aer code in this project to run new experiments, display bitstring probabilities, show the sampled calibration matrix, run a fresh noise sweep, generate a result-based recommendation, and export CSV / Markdown reports.

### Start the website on Windows (VS Code)

Open a terminal in the folder containing `main.py`, then run:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000` in your browser. Keep the VS Code terminal running while you use the site. Press `Ctrl+C` in the terminal to stop it.

If you have not generated the saved graphs/CSVs yet, run this once in a second terminal first:

```powershell
python main.py
```

### Website features

- **Experiment lab:** select Bell, GHZ3, or Grover2; choose gate, readout, or combined noise; set noise level, shots, and seed.
- **Fresh simulation:** the backend executes Qiskit Aer and samples actual measurement counts. The site does not fabricate new experiment values.
- **Measurement distribution:** raw, mitigated, and ideal probabilities by bitstring.
- **Calibration matrix:** sampled readout confusion matrix displayed as a heatmap.
- **Noise sensitivity:** runs a fresh sweep across the project's noise levels with Wilson 95% intervals.
- **Result-based assessment:** a transparent heuristic based on change in success probability and total variation distance; it is not a statistical-significance test.
- **Exports:** one-row experiment CSV and a Markdown report including settings, probabilities, intervals, counts, circuit metrics, recommendation, and limitations.
- **Saved analysis:** view the CSV results from `outputs/` and open the saved project graphs.

### API routes

- `GET /api/health` — backend status
- `GET /api/summary` — saved CSVs, circuit information, metadata
- `POST /api/run` — run one fresh experiment and calibration
- `POST /api/sweep` — run a fresh noise sensitivity sweep
- `POST /api/export/csv` — export the latest run as CSV
- `POST /api/export/report` — export a Markdown report

### Notes and limitations

- The backend uses **Qiskit Aer simulation only**. It does not connect to real IBM quantum hardware.
- The default interactive run uses 4,096 shots and 200 bootstrap resamples. Increase shots for more stable estimates, at the cost of runtime.
- Calibration-matrix inversion can amplify sampling noise. The mitigated bootstrap interval does not include uncertainty in the calibration matrix.
- A result-based recommendation is a heuristic and should be interpreted alongside confidence intervals, TVD, and repeated experiments.
- The frontend is served by FastAPI from `webapp/static`; no separate Node.js build is needed.

## Round 2 prototype additions

- **Mitigation Arena:** compares raw sampling, readout mitigation, simplified digital-noise-scaling zero-noise extrapolation (ZNE), and readout mitigation plus ZNE. The current ZNE demo scales the configured gate-noise parameters at factors 1, 2, and 3 and linearly extrapolates to zero. This is an educational approximation, not pulse-level gate folding; extrapolation may be unstable.
- **Noise Detective:** reports a heuristic diagnosis using sampled calibration-matrix confusion and residual distribution error. It is not a unique physical diagnosis and should be interpreted as an indicator.
- **Device profile:** generic configurable Qiskit Aer noise is available by default. Optional `IBM FakeManilaV2` mode uses a stored fake-backend noise snapshot through `qiskit-ibm-runtime`; it is still simulated data, not live IBM hardware, and supports at most five circuit qubits.
- **Larger GHZ circuits:** GHZ3, GHZ5, and GHZ8 are available. For 4+ qubits, calibration uses a tensor product of single-qubit calibration matrices to reduce the number of calibration experiments. This assumes independent readout errors and can be inaccurate when readout errors are correlated.
- **Bring your own circuit:** select OpenQASM 2.0, paste a circuit with 1–8 qubits, and enter expected successful bitstrings separated by commas. Mid-circuit measurement/classical control is not supported in this prototype. The Mitigation Arena currently supports built-in circuits only.
- **Plain-language explainer:** run-specific explanatory text is generated locally from the measured result; no external AI API key is required.

### Run the web prototype

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000`. The optional IBM fake-provider package is included in requirements; if it fails to install, generic Aer mode remains available. Do not describe fake-provider simulations as real-device measurements.

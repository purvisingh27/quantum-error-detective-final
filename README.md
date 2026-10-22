# 🔬 Quantum Error Detective

**An interactive quantum-noise analysis and error-mitigation laboratory built with Qiskit, Qiskit Aer, Python, and FastAPI.**

🌐 **Live Demo:** https://quantum-error-detective-final.onrender.com

💻 **Source Code:** https://github.com/purvisingh27/quantum-error-detective-final

## 🚀 Overview

Quantum Error Detective helps users explore how noise affects quantum circuits and compare techniques that attempt to reduce its impact. It provides an interactive environment for running simulations, inspecting measurement results, and understanding the limitations of error mitigation.

## ✨ Key Features

- **Quantum circuit experiments:** Explore Bell, GHZ, and Grover circuits, with support for larger GHZ circuits.
- **Noise simulation:** Investigate configurable gate and readout noise using Qiskit Aer.
- **Error mitigation:** Compare raw measurement results with readout-mitigated results.
- **Mitigation Arena:** Compare raw, readout mitigation, zero-noise extrapolation (ZNE), and combined approaches.
- **Noise sensitivity sweep:** Examine how changing noise levels affects experiment outcomes.
- **Noise Detective:** Explore heuristic explanations of possible noise contributions.
- **Statistical analysis:** Inspect success estimates, confidence intervals, and total variation distance (TVD).
- **Visualizations and exports:** Generate analysis graphs and export CSV data and reports.
- **Custom circuits:** Experiment with supported OpenQASM 2.0 inputs.
- **Optional fake-device simulation:** Explore a simulated IBM backend noise snapshot where supported.

## 🧰 Technology Stack

- Python
- Qiskit
- Qiskit Aer
- FastAPI
- NumPy and Pandas
- Matplotlib
- HTML, CSS, and JavaScript
- Optional `qiskit-ibm-runtime` integration

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/purvisingh27/quantum-error-detective-final.git
cd quantum-error-detective-final
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the web application

```bash
python -m uvicorn api:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## 🧪 How It Works

1. Select a supported quantum circuit and configure the experiment.
2. Run a baseline or noisy simulation.
3. Compare measurement distributions and success estimates.
4. Apply supported mitigation methods.
5. Inspect graphs, statistical metrics, and generated reports.

## ⚠️ Scientific Notes and Limitations

- Results are simulations and should not be presented as measurements from real quantum hardware.
- The optional IBM fake-backend mode uses a simulated backend, not a live IBM quantum processor.
- Noise models are illustrative and may not represent a particular physical device.
- Readout mitigation depends on calibration quality and assumptions about readout errors.
- Zero-noise extrapolation is an educational approximation in this project; results can be unstable and do not guarantee improved accuracy.
- A higher estimated success probability does not automatically mean a result is closer to the ideal distribution.
- Statistical estimates depend on shot counts, simulation settings, and sampling variability.

## 🎯 Project Goal

Make quantum noise, measurement errors, and error-mitigation trade-offs easier to explore through an accessible, interactive simulation tool.

## 👩‍💻 Built With

Created as a quantum-computing project for experimentation, visualization, and learning.

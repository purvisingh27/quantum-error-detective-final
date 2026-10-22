"""Readout error mitigation using a calibration (confusion) matrix.

Idea
----
1. Prepare every computational basis state |s> and measure it. Record how often
   each outcome m is seen. This gives a matrix A with A[m, s] = P(measure m | prepared s).
2. For any real experiment the measured distribution is  p_meas = A @ p_true.
3. Estimate the true distribution by inverting A:  p_est = A^-1 @ p_meas,
   then clip tiny negative values to 0 and renormalise.

Bit ordering: bitstring "q2 q1 q0" maps to integer index int(bitstring, 2),
so qubit i corresponds to bit i of the index.
"""

import numpy as np
from qiskit import QuantumCircuit

from noise import transpile_for_noise


def calibration_circuits(n: int):
    """One circuit per basis state, preparing |s> and measuring all qubits."""
    circuits = []
    for s in range(2 ** n):
        qc = QuantumCircuit(n, n, name=f"cal_{s:0{n}b}")
        for i in range(n):
            if (s >> i) & 1:
                qc.x(i)
        qc.measure(range(n), range(n))
        circuits.append(qc)
    return circuits


def counts_to_vector(counts: dict, n: int) -> np.ndarray:
    """Convert a Qiskit counts dict to a probability vector of length 2**n."""
    vec = np.zeros(2 ** n)
    for bitstring, c in counts.items():
        vec[int(bitstring.replace(" ", ""), 2)] += c
    total = vec.sum()
    return vec / total if total > 0 else vec


def build_calibration_matrix(backend, n: int, shots: int = 8192, seed: int = 0) -> np.ndarray:
    """Measure the confusion matrix A[m, s] on `backend` (which carries the noise)."""
    cals = [transpile_for_noise(c, backend, seed) for c in calibration_circuits(n)]
    result = backend.run(cals, shots=shots, seed_simulator=seed).result()
    A = np.zeros((2 ** n, 2 ** n))
    for s in range(2 ** n):
        A[:, s] = counts_to_vector(result.get_counts(s), n)
    return A


def mitigate_probabilities(p_measured: np.ndarray, A: np.ndarray, inverse: np.ndarray | None = None) -> np.ndarray:
    """Invert the calibration matrix, then project back onto a valid distribution.

    Pass a precomputed pseudoinverse when mitigating many bootstrap samples; this
    avoids repeating an expensive matrix decomposition for larger circuits.
    """
    inv = inverse if inverse is not None else np.linalg.pinv(A)
    p_est = inv @ p_measured
    p_est = np.clip(p_est, 0.0, None)
    total = p_est.sum()
    return p_est / total if total > 0 else p_measured


def success_from_vector(p: np.ndarray, expected, n: int) -> float:
    return float(sum(p[int(b, 2)] for b in expected))

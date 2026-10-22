"""Simulated noise models.

Three kinds of noise are studied separately and together:

  * gate noise     - depolarizing error after every physical gate
                     (two-qubit gates are noisier than one-qubit gates)
  * readout noise  - the measurement device sometimes reports the wrong bit
  * combined noise - both of the above

Circuits are transpiled to a fixed basis set so that the noise model applies to
the gates that a real device would actually execute.
"""

from qiskit import transpile
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

BASIS_GATES = ["rz", "sx", "x", "cx"]
TWO_QUBIT_FACTOR = 10.0   # CX error = 10 x single-qubit error
READOUT_ASYMMETRY = 1.5   # P(read 0 | state 1) = 1.5 x P(read 1 | state 0)


def build_noise_model(gate_error: float = 0.0, readout_error: float = 0.0):
    """Create an Aer noise model.

    gate_error:    1-qubit depolarizing probability (2-qubit = 10x, capped at 0.75)
    readout_error: P(measure 1 | prepared 0); P(measure 0 | prepared 1) is 1.5x
                   this value (capped at 0.5), mimicking T1 decay during readout.
    Returns None when both are zero (ideal simulation).
    """
    if gate_error <= 0 and readout_error <= 0:
        return None

    model = NoiseModel(basis_gates=BASIS_GATES)

    if gate_error > 0:
        p1 = min(gate_error, 0.75)
        p2 = min(gate_error * TWO_QUBIT_FACTOR, 0.75)
        model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
        model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])

    if readout_error > 0:
        p01 = min(readout_error, 0.5)
        p10 = min(readout_error * READOUT_ASYMMETRY, 0.5)
        model.add_all_qubit_readout_error(ReadoutError([[1 - p01, p01], [p10, 1 - p10]]))

    return model


def transpile_for_noise(circuit, backend, seed: int = 0):
    """Transpile to the basis gates that the noise model is attached to."""
    # `backend` is accepted for API symmetry but deliberately not passed to transpile():
    # combining it with basis_gates triggers a Qiskit warning and adds nothing here.
    return transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=seed,
    )

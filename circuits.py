"""Quantum circuits used in Quantum Error Detective.

Each builder returns a `CircuitSpec` holding:
  * the circuit WITHOUT measurements (so ideal probabilities can be computed
    exactly with a statevector),
  * the set of bitstrings that count as a "correct" outcome,
  * a short description.

Use `with_measurements(spec.circuit)` to get the version that is actually run.
"""

from dataclasses import dataclass, field

from qiskit import QuantumCircuit


@dataclass
class CircuitSpec:
    name: str
    circuit: QuantumCircuit
    expected: frozenset          # bitstrings counted as success
    description: str = ""
    n_qubits: int = field(init=False)

    def __post_init__(self):
        self.n_qubits = self.circuit.num_qubits


def bell_circuit() -> CircuitSpec:
    """2-qubit Bell state (|00> + |11>)/sqrt(2). Success = correlated outcome."""
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    return CircuitSpec(
        name="Bell",
        circuit=qc,
        expected=frozenset({"00", "11"}),
        description="Entangled pair; ideal outcomes are 00 and 11 (50/50).",
    )


def ghz_circuit(n: int = 3) -> CircuitSpec:
    """n-qubit GHZ state (|0...0> + |1...1>)/sqrt(2)."""
    qc = QuantumCircuit(n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return CircuitSpec(
        name=f"GHZ{n}",
        circuit=qc,
        expected=frozenset({"0" * n, "1" * n}),
        description=f"{n}-qubit entangled state; ideal outcomes are all-0 and all-1.",
    )


def grover_circuit() -> CircuitSpec:
    """2-qubit Grover search with the marked item |11>.

    For N=4 a single Grover iteration finds the marked item with probability 1.
    """
    qc = QuantumCircuit(2, name="grover2")
    # Uniform superposition
    qc.h([0, 1])
    # Oracle: phase-flip |11>
    qc.cz(0, 1)
    # Diffusion operator
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])
    return CircuitSpec(
        name="Grover2",
        circuit=qc,
        expected=frozenset({"11"}),
        description="2-qubit Grover search for |11>; ideal success probability is 1.",
    )


def with_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of `circuit` with every qubit i measured into clbit i."""
    n = circuit.num_qubits
    measured = QuantumCircuit(n, n, name=circuit.name)
    measured.compose(circuit, inplace=True)
    measured.measure(range(n), range(n))
    return measured


def all_circuits():
    # Keep the small reference circuits and add larger GHZ sizes for scaling experiments.
    return [bell_circuit(), ghz_circuit(3), grover_circuit(), ghz_circuit(5), ghz_circuit(8)]

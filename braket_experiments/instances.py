"""Deterministic Ising instances used by the Braket experiments.

Two instances are defined:

- ``reference_chain_model``: the repository's three-variable reference
  model (a two-edge chain). It appears in the QAOA reproducibility tests
  and its coupling graph matches a three-atom Rydberg chain, which makes
  it the natural instance for the QuEra Aquila experiment.

- ``build_random_ising``: a deterministic eleven-variable instance drawn
  from a seeded generator (seed 42, all pairs coupled). Eleven qubits is
  small enough for exact statevector simulation, large enough to be
  nontrivial, and fits comfortably on the IonQ Forte Enterprise device.
"""

from __future__ import annotations

import numpy as np

from ising_hamiltonian_converter import IsingHamiltonian

RANDOM_ISING_SEED = 42
RANDOM_ISING_VARIABLES = 11


def reference_chain_model() -> IsingHamiltonian:
    """The repository's three-variable reference Ising model."""

    return IsingHamiltonian(
        h={0: 0.5, 1: -0.8, 2: 0.3},
        J={(0, 1): -1.2, (1, 2): 0.9},
        offset=2.0,
    )


def chain5_model() -> IsingHamiltonian:
    """Five-variable nearest-neighbor chain (analog-device friendly)."""

    return IsingHamiltonian(
        h={0: 0.5, 1: -0.8, 2: 0.3, 3: -0.6, 4: 0.4},
        J={(0, 1): -1.2, (1, 2): 0.9, (2, 3): -0.7, (3, 4): 1.1},
        offset=2.0,
    )


def build_random_ising(
    num_variables: int = RANDOM_ISING_VARIABLES,
    seed: int = RANDOM_ISING_SEED,
) -> IsingHamiltonian:
    """Deterministic all-to-all Ising instance with a seeded generator."""

    if num_variables < 2:
        raise ValueError("num_variables must be >= 2")
    rng = np.random.default_rng(seed)
    h = {index: float(rng.uniform(-0.5, 0.5)) for index in range(num_variables)}
    J = {}
    for u in range(num_variables):
        for v in range(u + 1, num_variables):
            J[(u, v)] = float(rng.uniform(-1.0, 1.0))
    return IsingHamiltonian(h=h, J=J, offset=0.0)


INSTANCES = {
    "chain3": reference_chain_model,
    "chain5": chain5_model,
    "random11": build_random_ising,
}


def get_instance(name: str) -> IsingHamiltonian:
    if name not in INSTANCES:
        raise ValueError(f"unknown instance {name!r}")
    return INSTANCES[name]()

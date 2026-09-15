"""Braket experiment module for CCE-QOS.

Runs the repository's QAOA/Ising pipeline on Amazon Braket managed
simulators, on an IonQ gate-based QPU, and as an analog Rydberg program on
QuEra Aquila, following the amazon-braket skill guidance. See
``docs/BRAKET_EXPERIMENTS.md`` for the method, device ARNs, costs, and
evidence boundaries.
"""

from .instances import build_random_ising, reference_chain_model

__all__ = ["build_random_ising", "reference_chain_model"]

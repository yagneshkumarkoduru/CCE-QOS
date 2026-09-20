"""Workload generator contract tests (no pipeline execution)."""

import json
from pathlib import Path

from benchmarks.workload_generator import generate_workload, perturb_workload

ROOT = Path(__file__).resolve().parents[1]


def test_generated_workload_is_valid_and_deterministic():
    first = generate_workload(31, seed=123)
    second = generate_workload(31, seed=123)
    assert first == second
    assert len(first["nodes"]) == 31
    ids = [node["id"] for node in first["nodes"]]
    assert len(ids) == len(set(ids))
    known = set(ids)
    for node in first["nodes"]:
        for dep in node["dependencies"]:
            assert dep in known
        assert node["compute_cycles"] > 0
        assert node["input_size"] >= 32.0
        assert node["output_size"] >= 32.0


def test_generated_workload_seed_changes_instance():
    assert generate_workload(31, seed=1) != generate_workload(31, seed=2)


def test_perturbation_preserves_topology_and_bounds():
    reference = json.loads((ROOT / "example_workload.json").read_text(encoding="utf-8"))
    perturbed = perturb_workload(reference, seed=7)
    assert [node["id"] for node in perturbed["nodes"]] == [node["id"] for node in reference["nodes"]]
    for ref, new in zip(reference["nodes"], perturbed["nodes"]):
        assert new["dependencies"] == ref["dependencies"]
        assert 0.5 * ref["compute_cycles"] <= new["compute_cycles"] <= 1.5 * ref["compute_cycles"]
        assert 0.5 * ref["input_size"] <= new["input_size"] <= 1.5 * ref["input_size"]

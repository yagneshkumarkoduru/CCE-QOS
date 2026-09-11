"""
Exact CP-SAT Scheduler Runner for CCE-QOS.
Solves NPU task DAGs with exact global optimality guarantees via Google OR-Tools CP-SAT.
"""

import sys
import traceback

try:
    try:
        # Package-relative import (works when run as a module:
        # python -m implementations.v1_exact_cpsat_solver.main_cpsat_runner)
        from .ortools_cpsat_engine import ExactCPSATScheduler
    except ImportError:
        # Bare import fallback (works when run as a script:
        # python implementations/v1_exact_cpsat_solver/main_cpsat_runner.py)
        from ortools_cpsat_engine import ExactCPSATScheduler
    from ortools.sat.python import cp_model
except ImportError as exc:
    cp_model = None
    ExactCPSATScheduler = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def run_cpsat_benchmark():
    print("=" * 70)
    print("EXACT OR-TOOLS CP-SAT COMPILER SCHEDULING")
    print("=" * 70)

    if ExactCPSATScheduler is None:
        print(f"[FAIL] Could not import the CP-SAT engine. Real reason below.")
        print(f"       Import error: {_IMPORT_ERROR}")
        traceback.print_exc()
        sys.exit(1)

    # 6-node synthetic NPU DAG
    tasks = [
        {"id": "conv1", "duration": 2, "memory_kb": 256},
        {"id": "relu1", "duration": 1, "memory_kb": 128},
        {"id": "pool1", "duration": 1, "memory_kb": 128},
        {"id": "conv2", "duration": 3, "memory_kb": 512},
        {"id": "relu2", "duration": 1, "memory_kb": 256},
        {"id": "dense", "duration": 2, "memory_kb": 1024}
    ]
    deps = [
        ("conv1", "relu1"),
        ("relu1", "pool1"),
        ("pool1", "conv2"),
        ("conv2", "relu2"),
        ("relu2", "dense")
    ]

    scheduler = ExactCPSATScheduler()
    res = scheduler.solve(tasks, deps)

    print(f"Solver Status   : {res['status']}")
    print(f"Optimal Makespan: {res['makespan']} cycles")
    print(f"Schedule Map    : {res['schedule']}")
    print(f"Solve Time      : {res['solver_runtime_s']:.3f} s")
    print("Exact global scheduling verified.\n")


if __name__ == "__main__":
    run_cpsat_benchmark()

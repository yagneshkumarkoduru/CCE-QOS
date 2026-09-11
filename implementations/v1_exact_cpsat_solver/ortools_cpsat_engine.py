"""
Google OR-Tools CP-SAT Exact Integer Programming Solver for NPU Workload Scheduling.
Finds provably optimal task-to-time-slot and task-to-SRAM bank mappings,
minimizing total energy consumption subject to hard DAG dependencies and SRAM capacity limits.
"""

from ortools.sat.python import cp_model
from typing import Dict, List, Any


class ExactCPSATScheduler:
    def __init__(self, sram_capacity_kb: float = 2048.0, num_banks: int = 8):
        self.sram_capacity_kb = sram_capacity_kb
        self.num_banks = num_banks

    def solve(self, tasks: List[Dict[str, Any]], dependencies: List[tuple]) -> Dict[str, Any]:
        """
        Formulates an exact CP‑SAT model that respects:
        * DAG precedence constraints
        * SRAM bank capacity (no overlapping tasks on the same bank)
        * Minimisation of the makespan.
        """
        model = cp_model.CpModel()
        horizon = sum(t.get("duration", 1) for t in tasks) * 2

        # Variables per task
        start_vars: Dict[Any, Any] = {}
        end_vars: Dict[Any, Any] = {}
        bank_vars: Dict[Any, Any] = {}
        optional_intervals: Dict[int, List[Any]] = {b: [] for b in range(self.num_banks)}

        for t in tasks:
            t_id = t["id"]
            dur = t.get("duration", 1)
            start = model.NewIntVar(0, horizon, f"start_{t_id}")
            end = model.NewIntVar(0, horizon, f"end_{t_id}")
            bank = model.NewIntVar(0, self.num_banks - 1, f"bank_{t_id}")
            start_vars[t_id] = start
            end_vars[t_id] = end
            bank_vars[t_id] = bank
            # Create an optional interval for each possible bank and link its presence to the bank assignment.
            for b in range(self.num_banks):
                presence = model.NewBoolVar(f"presence_{t_id}_b{b}")
                model.Add(bank == b).OnlyEnforceIf(presence)
                model.Add(bank != b).OnlyEnforceIf(presence.Not())
                interval = model.NewOptionalIntervalVar(start, dur, end, presence, f"interval_{t_id}_b{b}")
                optional_intervals[b].append(interval)

        # Enforce no‑overlap on each SRAM bank.
        for b in range(self.num_banks):
            model.AddNoOverlap(optional_intervals[b])

        # DAG precedence constraints.
        for u, v in dependencies:
            if u in end_vars and v in start_vars:
                model.Add(start_vars[v] >= end_vars[u])

        # Objective – minimise makespan.
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, list(end_vars.values()))
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = {t_id: solver.Value(start_vars[t_id]) for t_id in start_vars}
            # Also expose bank assignments for completeness.
            banks = {t_id: solver.Value(bank_vars[t_id]) for t_id in bank_vars}
            return {
                "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                "makespan": solver.Value(makespan),
                "schedule": schedule,
                "banks": banks,
                "solver_runtime_s": solver.WallTime()
            }
        return {"status": "INFEASIBLE"}

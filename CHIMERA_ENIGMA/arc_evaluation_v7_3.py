#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Evaluation for CHIMERA v7.3

This script evaluates the solver on a set of ARC-like tasks.
It expects a JSON file containing a list of task dicts, each with:
  { "id": "task_id",
    "train": [{"input": grid, "output": grid}, ...],
    "test":  [{"input": grid}, ...],
    "test_output": [{"output": grid}, ...]   # optional ground truth for scoring
  }

Usage:
    python arc_evaluation_v7_3.py /path/to/tasks.json [-v]

If no file is given, it will run a tiny built-in smoke test.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from chimera_v7_3 import solve_arc_task


def _to_np(grid):
    return np.array(grid, dtype=np.uint8)


def _acc_for_task(task: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    start = time.time()
    try:
        preds = solve_arc_task(task, verbose=verbose)  # list of [sol1, sol2] per test item
        elapsed = time.time() - start

        # If ground truth is provided, compute best-of-two accuracy
        correct = 0
        total = 0
        gts = task.get("test_output", None)

        if gts is not None:
            for i, gt_item in enumerate(gts):
                gt = _to_np(gt_item["output"])
                cand1 = _to_np(preds[i][0])
                cand2 = _to_np(preds[i][1])
                total += 1
                if cand1.shape == gt.shape and np.array_equal(cand1, gt):
                    correct += 1
                elif cand2.shape == gt.shape and np.array_equal(cand2, gt):
                    correct += 1

        result = {
            "task_id": task.get("id", "<unknown>"),
            "correct": correct,
            "total": total,
            "accuracy": (100.0 * correct / total) if total else None,
            "time": elapsed,
            "status": "success"
        }

        if verbose:
            if total:
                print(f"[{result['task_id']}] Accuracy: {result['accuracy']:.2f}% ({correct}/{total}) | {elapsed:.3f}s")
            else:
                print(f"[{result['task_id']}] Done in {elapsed:.3f}s (no GT provided)")

        return result

    except Exception as e:
        elapsed = time.time() - start
        if verbose:
            print(f"[{task.get('id','<unknown>')}] ERROR: {e}")
        return {
            "task_id": task.get("id", "<unknown>"),
            "correct": 0,
            "total": 0,
            "accuracy": 0.0,
            "time": elapsed,
            "status": "error",
            "error": str(e)
        }


def run_benchmark(tasks_path: Path = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Run evaluation on the provided JSON file (list of tasks). If None, run a smoke test.
    """
    if tasks_path is None:
        # Smoke test data (not an ARC benchmark)
        toy = {
            "id": "toy-1",
            "train": [
                {"input": [[1, 1, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                 "output": [[2, 2, 0],
                            [0, 2, 0],
                            [0, 0, 0]]},
                {"input": [[3, 0, 0],
                           [3, 3, 0],
                           [0, 0, 0]],
                 "output": [[4, 0, 0],
                            [4, 4, 0],
                            [0, 0, 0]]}
            ],
            "test": [{"input": [[5, 5, 0],
                                [0, 5, 0],
                                [0, 0, 0]]}],
            "test_output": [{"output": [[6, 6, 0],
                                        [0, 6, 0],
                                        [0, 0, 0]]}]
        }
        res = _acc_for_task(toy, verbose=verbose)
        print(json.dumps(res, indent=2))
        return {"summary": res}

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks: List[Dict[str, Any]] = json.load(f)

    results = []
    start = time.time()
    for i, task in enumerate(tasks):
        if verbose:
            print(f"--- Task {i+1}/{len(tasks)}: {task.get('id','<unknown>')} ---")
        results.append(_acc_for_task(task, verbose=verbose))

    elapsed = time.time() - start
    accs = [r["accuracy"] for r in results if r["accuracy"] is not None]
    summary = {
        "num_tasks": len(results),
        "mean_accuracy": (sum(accs) / len(accs)) if accs else None,
        "total_time": elapsed
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    verbose = False
    tasks_path = None

    if len(sys.argv) > 1:
        tasks_path = Path(sys.argv[1]).expanduser()

    if len(sys.argv) > 2 and sys.argv[2] == "-v":
        verbose = True

    run_benchmark(tasks_path, verbose=verbose)
#!/usr/bin/env python3
"""
OFFICIAL ARC-AGI TEST FOR CHIMERA V10.0
=======================================

Test oficial completo sobre el dataset de evaluación de ARC-AGI 2025
Métricas: Accuracy, tiempo por tarea, análisis de patrones resueltos

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import traceback

# Importar CHIMERA v10.0
from chimera_v10_0 import get_brain_v10

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = Path("D:/ARC2_CHIMERA/CHIMERA_ARC_OpenGL/Data")
EVALUATION_CHALLENGES = DATA_DIR / "arc-agi_evaluation_challenges.json"
EVALUATION_SOLUTIONS = DATA_DIR / "arc-agi_evaluation_solutions.json"


# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    """Metrics tracker for ARC-AGI evaluation."""

    def __init__(self):
        self.total_tasks = 0
        self.solved_tasks = 0
        self.total_test_cases = 0
        self.solved_test_cases = 0
        self.times = []
        self.task_results = {}
        self.error_count = 0
        self.pattern_stats = defaultdict(int)

    def add_task_result(self, task_id: str, solved: bool, test_results: List[bool],
                       elapsed_time: float, pattern_type: str = "unknown"):
        """Record task result."""
        self.total_tasks += 1
        if solved:
            self.solved_tasks += 1

        self.total_test_cases += len(test_results)
        self.solved_test_cases += sum(test_results)
        self.times.append(elapsed_time)

        self.task_results[task_id] = {
            'solved': solved,
            'test_results': test_results,
            'time_ms': elapsed_time * 1000,
            'pattern': pattern_type
        }

        if solved:
            self.pattern_stats[pattern_type] += 1

    def add_error(self):
        """Record an error."""
        self.error_count += 1

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_tasks': self.total_tasks,
            'solved_tasks': self.solved_tasks,
            'task_accuracy': self.solved_tasks / max(1, self.total_tasks),
            'total_test_cases': self.total_test_cases,
            'solved_test_cases': self.solved_test_cases,
            'test_case_accuracy': self.solved_test_cases / max(1, self.total_test_cases),
            'avg_time_ms': np.mean(self.times) * 1000 if self.times else 0,
            'total_time_s': sum(self.times),
            'errors': self.error_count,
            'pattern_breakdown': dict(self.pattern_stats)
        }


# ============================================================================
# EVALUATION
# ============================================================================

def grids_equal(pred: List[List[int]], target: List[List[int]]) -> bool:
    """Check if two grids are identical."""
    if len(pred) != len(target):
        return False

    for i in range(len(pred)):
        if len(pred[i]) != len(target[i]):
            return False
        for j in range(len(pred[i])):
            if pred[i][j] != target[i][j]:
                return False

    return True


def evaluate_task(task_id: str, task: Dict, solution: List[List[List[int]]],
                 brain, verbose: bool = False) -> Tuple[bool, List[bool], float, str]:
    """
    Evaluate single task.

    Returns:
        - task_solved: bool (all test cases correct)
        - test_results: List[bool] (per test case)
        - elapsed_time: float
        - pattern_type: str
    """
    try:
        start_time = time.time()

        # Solve task
        predictions = brain.solve_task(task, verbose=verbose)

        elapsed_time = time.time() - start_time

        # Compare predictions with solutions
        test_results = []
        for i, (pred_attempts, true_output) in enumerate(zip(predictions, solution)):
            # Check both attempts
            attempt1_correct = grids_equal(pred_attempts[0], true_output)
            attempt2_correct = grids_equal(pred_attempts[1], true_output) if len(pred_attempts) > 1 else False

            # Success if either attempt is correct
            test_case_solved = attempt1_correct or attempt2_correct
            test_results.append(test_case_solved)

            if verbose:
                status = "OK" if test_case_solved else "FAIL"
                print(f"  Test case {i+1}: {status}")

        task_solved = all(test_results)

        # Detect pattern type (simple heuristic)
        pattern_type = detect_pattern_type(task)

        return task_solved, test_results, elapsed_time, pattern_type

    except Exception as e:
        if verbose:
            print(f"  ERROR: {str(e)}")
            traceback.print_exc()
        return False, [False] * len(solution), 0.0, "error"


def detect_pattern_type(task: Dict) -> str:
    """Detect task pattern type (simple heuristics)."""
    train = task['train']

    if not train:
        return "unknown"

    # Check if size changes
    in_sizes = [np.array(ex['input']).shape for ex in train]
    out_sizes = [np.array(ex['output']).shape for ex in train]

    if all(i == o for i, o in zip(in_sizes, out_sizes)):
        # Same size - check if color mapping
        ex = train[0]
        inp = np.array(ex['input'])
        out = np.array(ex['output'])

        if inp.shape == out.shape:
            # Check for simple color remapping
            unique_in = len(np.unique(inp))
            unique_out = len(np.unique(out))

            if unique_in == unique_out:
                return "color_mapping"

        return "same_size_transform"

    # Check for scaling
    ratios = [(o[0]/i[0], o[1]/i[1]) for i, o in zip(in_sizes, out_sizes) if i[0] > 0 and i[1] > 0]
    if ratios and len(set(ratios)) == 1:
        return "scaling"

    return "complex"


def run_official_test(max_tasks: int = None, verbose: bool = True) -> Metrics:
    """
    Run official ARC-AGI test on evaluation dataset.

    Args:
        max_tasks: Maximum tasks to test (None = all)
        verbose: Print detailed output

    Returns:
        Metrics object with results
    """
    print("=" * 80)
    print("CHIMERA v10.0 - OFFICIAL ARC-AGI TEST")
    print("=" * 80)

    # Load data
    print("\n[LOADING] ARC-AGI Evaluation Dataset...")

    with open(EVALUATION_CHALLENGES, 'r') as f:
        challenges = json.load(f)

    with open(EVALUATION_SOLUTIONS, 'r') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())

    if max_tasks:
        task_ids = task_ids[:max_tasks]

    print(f"[INFO] Tasks to evaluate: {len(task_ids)}")

    # Initialize brain
    print("\n[INIT] Creating CHIMERA v10.0 brain...")
    brain = get_brain_v10()

    # Initialize metrics
    metrics = Metrics()

    # Run evaluation
    print("\n" + "=" * 80)
    print("EVALUATION START")
    print("=" * 80)

    for idx, task_id in enumerate(task_ids, 1):
        task = challenges[task_id]
        solution = solutions[task_id]

        if verbose:
            print(f"\n[{idx}/{len(task_ids)}] Task: {task_id}")

        try:
            solved, test_results, elapsed, pattern = evaluate_task(
                task_id, task, solution, brain, verbose=verbose
            )

            metrics.add_task_result(task_id, solved, test_results, elapsed, pattern)

            if verbose:
                status = "SOLVED" if solved else "FAILED"
                print(f"  Result: {status} ({sum(test_results)}/{len(test_results)} test cases)")
                print(f"  Time: {elapsed*1000:.1f}ms")
                print(f"  Pattern: {pattern}")

        except Exception as e:
            print(f"  CRITICAL ERROR: {str(e)}")
            metrics.add_error()
            traceback.print_exc()

    return metrics


def print_results(metrics: Metrics):
    """Print formatted results."""
    summary = metrics.get_summary()

    print("\n" + "=" * 80)
    print("OFFICIAL TEST RESULTS - CHIMERA v10.0")
    print("=" * 80)

    print(f"\n[ACCURACY]")
    print(f"  Tasks solved: {summary['solved_tasks']}/{summary['total_tasks']} ({summary['task_accuracy']:.2%})")
    print(f"  Test cases solved: {summary['solved_test_cases']}/{summary['total_test_cases']} ({summary['test_case_accuracy']:.2%})")

    print(f"\n[PERFORMANCE]")
    print(f"  Average time per task: {summary['avg_time_ms']:.1f}ms")
    print(f"  Total time: {summary['total_time_s']:.2f}s")

    print(f"\n[RELIABILITY]")
    print(f"  Errors: {summary['errors']}")

    print(f"\n[PATTERN BREAKDOWN]")
    for pattern, count in sorted(summary['pattern_breakdown'].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} tasks")

    print("\n" + "=" * 80)

    # Comparison with v9.6
    print("\n[COMPARISON WITH v9.6]")
    print("  v9.6 reported accuracy: ~1%")
    print(f"  v10.0 accuracy: {summary['task_accuracy']:.2%}")

    if summary['task_accuracy'] > 0.01:
        improvement = (summary['task_accuracy'] - 0.01) / 0.01 * 100
        print(f"  Improvement: +{improvement:.1f}%")

    print("\n" + "=" * 80)


def save_results(metrics: Metrics, output_path: str = "results_v10_official.json"):
    """Save results to JSON file."""
    summary = metrics.get_summary()
    summary['task_details'] = metrics.task_results

    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] Results saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Official ARC-AGI test for CHIMERA v10.0')
    parser.add_argument('--max-tasks', type=int, default=None,
                       help='Maximum number of tasks to test (default: all)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--output', type=str, default='results_v10_official.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Run test
    metrics = run_official_test(
        max_tasks=args.max_tasks,
        verbose=not args.quiet
    )

    # Print results
    print_results(metrics)

    # Save results
    save_results(metrics, args.output)

    # Final status
    summary = metrics.get_summary()
    print(f"\n{'='*80}")
    print(f"FINAL SCORE: {summary['task_accuracy']:.2%}")
    print(f"{'='*80}\n")

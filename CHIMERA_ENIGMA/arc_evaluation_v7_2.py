#!/usr/bin/env python3
"""
ARC Evaluation for CHIMERA v7.2

Evaluates the fixed unified GPU architecture on ARC-AGI2 training dataset.

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import json
import time
from pathlib import Path
from chimera_v7_2 import solve_arc_task
import numpy as np


def load_arc_tasks(challenges_file: str = "data/arc-agi_training_challenges.json",
                   solutions_file: str = "data/arc-agi_training_solutions.json"):
    """Load ARC training tasks with solutions"""
    tasks = []

    challenges_path = Path(challenges_file)
    solutions_path = Path(solutions_file)

    if not challenges_path.exists():
        print(f"[ERROR] Challenges file not found: {challenges_file}")
        return tasks

    if not solutions_path.exists():
        print(f"[ERROR] Solutions file not found: {solutions_file}")
        return tasks

    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    # Merge challenges and solutions
    for task_id, task_data in challenges.items():
        task_data['id'] = task_id

        # Add solutions to test cases
        if task_id in solutions:
            for i, test_case in enumerate(task_data['test']):
                if i < len(solutions[task_id]):
                    test_case['output'] = solutions[task_id][i]

        tasks.append(task_data)

    return tasks


def evaluate_task(task: dict, verbose: bool = False) -> dict:
    """Evaluate CHIMERA v7.2 on a single task"""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Task: {task['id']}")
        print(f"{'='*80}")

    start_time = time.time()

    try:
        # Solve task
        predictions = solve_arc_task(task, verbose=verbose)

        elapsed = time.time() - start_time

        # Check correctness for each test case
        correct = 0
        total = len(task['test'])

        for test_idx, test_case in enumerate(task['test']):
            expected_output = test_case['output']

            # Check both attempts
            for attempt_idx, attempt in enumerate(predictions[test_idx]):
                if np.array_equal(np.array(attempt), np.array(expected_output)):
                    correct += 1
                    if verbose:
                        print(f"  Test {test_idx} Attempt {attempt_idx}: ✓ CORRECT")
                    break
            else:
                if verbose:
                    print(f"  Test {test_idx}: ✗ INCORRECT")

        accuracy = (correct / total) * 100 if total > 0 else 0

        result = {
            'task_id': task['id'],
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'time': elapsed,
            'status': 'success'
        }

        if verbose:
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
            print(f"  Time: {elapsed:.3f}s")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ERROR: {e}")
        return {
            'task_id': task['id'],
            'correct': 0,
            'total': len(task['test']),
            'accuracy': 0,
            'time': elapsed,
            'status': 'error',
            'error': str(e)
        }


def run_benchmark(num_tasks: int = 100, verbose: bool = False):
    """Run benchmark on ARC training set"""

    print("=" * 80)
    print("CHIMERA v7.2 - ARC-AGI2 BENCHMARK")
    print("=" * 80)
    print("Architecture: Unified GPU with Semantic Memory")
    print("Fixes: 5 critical weaknesses from v7.1")
    print("=" * 80)

    # Load tasks
    tasks = load_arc_tasks()

    if not tasks:
        print("[ERROR] No tasks loaded!")
        return

    print(f"\nLoaded {len(tasks)} tasks")
    print(f"Evaluating first {num_tasks} tasks...")
    print()

    # Evaluate
    results = []
    total_correct = 0
    total_tests = 0
    total_time = 0

    for i, task in enumerate(tasks[:num_tasks]):
        result = evaluate_task(task, verbose=verbose)
        results.append(result)

        total_correct += result['correct']
        total_tests += result['total']
        total_time += result['time']

        # Progress update
        if (i + 1) % 10 == 0:
            current_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
            print(f"Progress: {i+1}/{num_tasks} tasks | "
                  f"Accuracy: {current_acc:.2f}% | "
                  f"Avg time: {total_time/(i+1):.3f}s/task")

    # Final results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    avg_time = total_time / len(results) if results else 0

    print(f"Tasks evaluated: {len(results)}")
    print(f"Test cases: {total_tests}")
    print(f"Correct predictions: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Average time per task: {avg_time:.3f}s")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 80)

    # Save results
    output_file = f"results/benchmark_v7_2_{int(time.time())}.json"
    Path("results").mkdir(exist_ok=True)

    summary = {
        'version': 'v7.2',
        'tasks_evaluated': len(results),
        'total_tests': total_tests,
        'correct_predictions': total_correct,
        'overall_accuracy': overall_accuracy,
        'avg_time_per_task': avg_time,
        'total_time': total_time,
        'results': results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == "__main__":
    import sys

    # Parse arguments
    num_tasks = 100
    verbose = False

    if len(sys.argv) > 1:
        num_tasks = int(sys.argv[1])

    if len(sys.argv) > 2 and sys.argv[2] == '-v':
        verbose = True

    # Run benchmark
    run_benchmark(num_tasks=num_tasks, verbose=verbose)

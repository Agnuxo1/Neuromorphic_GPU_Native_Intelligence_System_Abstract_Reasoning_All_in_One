#!/usr/bin/env python3
"""
Benchmark for CHIMERA v7.4 (Phases 1-4)
"""

import json
import sys
from pathlib import Path
from chimera_v7_4 import solve_arc_task
import numpy as np
import time

def load_arc_tasks(num_tasks=20):
    """Load ARC training tasks with solutions"""
    challenges_file = "data/arc-agi_training_challenges.json"
    solutions_file = "data/arc-agi_training_solutions.json"

    with open(challenges_file, 'r') as f:
        challenges = json.load(f)

    with open(solutions_file, 'r') as f:
        solutions = json.load(f)

    tasks = []
    for task_id, task_data in list(challenges.items())[:num_tasks]:
        task = {
            'id': task_id,
            'train': task_data['train'],
            'test': task_data['test']
        }

        # Add ground truth for evaluation
        task['test_outputs'] = []
        if task_id in solutions:
            for sol in solutions[task_id]:
                task['test_outputs'].append(sol)

        tasks.append(task)

    return tasks

def evaluate_task(task, verbose=False):
    """Evaluate single task"""
    start = time.time()

    try:
        predictions = solve_arc_task(task, verbose=verbose)
        elapsed = time.time() - start

        # Check correctness
        correct = 0
        total = len(task['test_outputs'])

        for i, gt in enumerate(task['test_outputs']):
            gt_array = np.array(gt, dtype=np.uint8)

            # Check both attempts
            for attempt in predictions[i]:
                pred = np.array(attempt, dtype=np.uint8)
                if np.array_equal(pred, gt_array):
                    correct += 1
                    if verbose:
                        print(f"  Test {i}: ✓ CORRECT")
                    break
            else:
                if verbose:
                    print(f"  Test {i}: ✗ INCORRECT")

        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            'task_id': task['id'],
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'time': elapsed,
            'status': 'success'
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'task_id': task['id'],
            'correct': 0,
            'total': len(task.get('test_outputs', [])),
            'accuracy': 0,
            'time': elapsed,
            'status': 'error',
            'error': str(e)
        }

def run_benchmark(num_tasks=20, verbose=False):
    """Run benchmark"""
    print("=" * 80)
    print("CHIMERA v7.4 - ARC-AGI2 BENCHMARK (Phases 1-4)")
    print("=" * 80)
    print("Improvements over v7.2:")
    print("  + Massive candidate generation (beam K=64)")
    print("  + Supervised ranking (Hamming/IoU loss)")
    print("  + Hard constraints (palette, validity)")
    print("  + Two diverse attempts")
    print("=" * 80)

    # Load tasks
    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks")
    print(f"Evaluating {num_tasks} tasks...\n")

    # Evaluate
    results = []
    total_correct = 0
    total_tests = 0
    total_time = 0

    for i, task in enumerate(tasks):
        if not verbose:
            print(f"\nTask {i+1}/{num_tasks}: {task['id']}")

        result = evaluate_task(task, verbose=verbose)
        results.append(result)

        total_correct += result['correct']
        total_tests += result['total']
        total_time += result['time']

        if not verbose:
            print(f"  Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']}) | Time: {result['time']:.3f}s")

        # Progress update every 10 tasks
        if (i + 1) % 10 == 0:
            current_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
            print(f"\nProgress: {i+1}/{num_tasks} tasks | "
                  f"Cumulative Accuracy: {current_acc:.2f}% | "
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

    # Compare with v7.2
    print("\nComparison with v7.2:")
    print(f"  v7.2: 1.92% accuracy, 0.023s/task")
    print(f"  v7.4: {overall_accuracy:.2f}% accuracy, {avg_time:.3f}s/task")
    if overall_accuracy > 1.92:
        improvement = overall_accuracy - 1.92
        print(f"  Improvement: +{improvement:.2f} pp ✓")
    else:
        print(f"  No improvement yet")
    print("=" * 80)

    # Save results
    output_file = f"results/benchmark_v7_4_{int(time.time())}.json"
    Path("results").mkdir(exist_ok=True)

    summary = {
        'version': 'v7.4',
        'phases': '1-4 (quick wins)',
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
    num_tasks = 20
    verbose = False

    if len(sys.argv) > 1:
        num_tasks = int(sys.argv[1])

    if len(sys.argv) > 2 and sys.argv[2] == '-v':
        verbose = True

    run_benchmark(num_tasks=num_tasks, verbose=verbose)

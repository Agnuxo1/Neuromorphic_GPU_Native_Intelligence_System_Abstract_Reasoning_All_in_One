#!/usr/bin/env python3
"""
Benchmark CHIMERA ARC Unified (TRUE GPU architecture)

This is the REAL test - GPU renders solutions instead of calculating.
"""

import json
import time
import numpy as np
import sys

from chimera_arc_unified import solve_arc_task

def load_arc_tasks(num_tasks=100):
    """Load official ARC-AGI2 training tasks"""
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

def main():
    if len(sys.argv) > 1:
        num_tasks = int(sys.argv[1])
    else:
        num_tasks = 100

    print("="*80)
    print("CHIMERA ARC UNIFIED - BENCHMARK")
    print("="*80)
    print("TRUE GPU Architecture: GPU renders (not calculates)")
    print("="*80)

    # Load tasks
    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks from official training set\n")

    # Benchmark
    times = []
    correct = 0
    total = 0
    results = []

    for i, task in enumerate(tasks):
        start = time.time()

        try:
            predictions = solve_arc_task(task, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)

            # Check correctness
            task_correct = False
            for j, gt in enumerate(task['test_outputs']):
                gt_array = np.array(gt, dtype=np.uint8)
                total += 1

                # Check both attempts
                for attempt in predictions[j]:
                    pred = np.array(attempt, dtype=np.uint8)
                    if np.array_equal(pred, gt_array):
                        correct += 1
                        task_correct = True
                        break

            status = "OK" if task_correct else "FAIL"
            results.append({
                'task_id': task['id'],
                'time': elapsed,
                'correct': task_correct
            })

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            total += len(task.get('test_outputs', []))
            status = "ERROR"
            results.append({
                'task_id': task['id'],
                'time': elapsed,
                'correct': False,
                'error': str(e)
            })

        if (i + 1) % 10 == 0:
            avg_time = np.mean(times)
            acc = (correct / total * 100) if total > 0 else 0
            print(f"  Progress: {i+1}/{len(tasks)} tasks | Avg time: {avg_time:.6f}s | Accuracy: {acc:.2f}%")

    # Final results
    avg_time = np.mean(times)
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Tasks: {len(tasks)}")
    print(f"Avg time per task: {avg_time:.6f}s")
    print(f"Total time: {sum(times):.3f}s")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*80)

    # Save results
    output = {
        'num_tasks': len(tasks),
        'avg_time': avg_time,
        'total_time': sum(times),
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }

    timestamp = int(time.time())
    output_file = f"results/arc_unified_benchmark_{num_tasks}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

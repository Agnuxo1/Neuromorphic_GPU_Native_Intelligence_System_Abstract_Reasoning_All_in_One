#!/usr/bin/env python3
"""
Benchmark GPU vs CPU to demonstrate actual speedup

Compares:
- v7.2 (CPU NumPy) vs chimera_gpu_final (GPU Shaders)
"""

import json
import time
import numpy as np
from chimera_v7_2 import solve_arc_task as solve_cpu
from chimera_gpu_final import solve_arc_task as solve_gpu

def load_arc_tasks(num_tasks=50):
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

def benchmark_solver(solve_fn, tasks, name):
    """Benchmark a solver"""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {name}")
    print(f"{'='*80}")

    times = []
    correct = 0
    total = 0

    for i, task in enumerate(tasks):
        start = time.time()

        try:
            predictions = solve_fn(task, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)

            # Check correctness
            for j, gt in enumerate(task['test_outputs']):
                gt_array = np.array(gt, dtype=np.uint8)
                total += 1

                # Check both attempts
                for attempt in predictions[j]:
                    pred = np.array(attempt, dtype=np.uint8)
                    if np.array_equal(pred, gt_array):
                        correct += 1
                        break

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            total += len(task.get('test_outputs', []))
            print(f"  Task {i+1}: ERROR - {e}")

        if (i + 1) % 10 == 0:
            avg_time = np.mean(times)
            print(f"  Progress: {i+1}/{len(tasks)} tasks, avg time: {avg_time:.6f}s")

    avg_time = np.mean(times)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nResults:")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Avg time per task: {avg_time:.6f}s")
    print(f"  Total time: {sum(times):.3f}s")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return {
        'name': name,
        'avg_time': avg_time,
        'total_time': sum(times),
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'times': times
    }

def main():
    print("="*80)
    print("CHIMERA GPU SPEEDUP BENCHMARK")
    print("="*80)
    print("Comparing CPU (v7.2) vs GPU (chimera_gpu_final)")
    print("="*80)

    # Load tasks
    num_tasks = 50
    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks from official training set")

    # Benchmark CPU
    cpu_results = benchmark_solver(solve_cpu, tasks, "CPU (v7.2 - NumPy)")

    # Benchmark GPU
    gpu_results = benchmark_solver(solve_gpu, tasks, "GPU (chimera_gpu_final - Shaders)")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON: CPU vs GPU")
    print("="*80)
    print(f"CPU (v7.2):              {cpu_results['avg_time']:.6f}s per task")
    print(f"GPU (chimera_gpu_final): {gpu_results['avg_time']:.6f}s per task")
    print(f"\nSpeedup: {cpu_results['avg_time'] / gpu_results['avg_time']:.2f}x")
    print(f"\nAccuracy:")
    print(f"  CPU: {cpu_results['accuracy']:.2f}%")
    print(f"  GPU: {gpu_results['accuracy']:.2f}%")
    print("="*80)

    # Save results
    output = {
        'cpu': cpu_results,
        'gpu': gpu_results,
        'speedup': cpu_results['avg_time'] / gpu_results['avg_time'],
        'num_tasks': num_tasks
    }

    with open(f"results/gpu_speedup_benchmark_{int(time.time())}.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/gpu_speedup_benchmark_{int(time.time())}.json")

if __name__ == "__main__":
    main()

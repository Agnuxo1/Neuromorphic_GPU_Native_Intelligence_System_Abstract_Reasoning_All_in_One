#!/usr/bin/env python3
"""
Benchmark CHIMERA Living Brain

The brain lives permanently - context never destroyed.
This is the TRUE architecture from the paper.
"""

import json
import time
import numpy as np
import sys

# Import the living brain
from chimera_living_brain import get_brain

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
    print("CHIMERA LIVING BRAIN - BENCHMARK")
    print("="*80)
    print("Brain lives permanently in GPU memory")
    print("State and memory persist across ALL tasks")
    print("="*80)

    # Load tasks
    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks from official training set")

    # Get the living brain (created once, lives forever)
    print("\nAwakening the brain...")
    brain = get_brain()
    print(f"Brain awakened. GPU: {brain.ctx.info['GL_RENDERER']}\n")

    # Benchmark
    times = []
    correct = 0
    total = 0
    results = []

    for i, task in enumerate(tasks):
        start = time.time()

        try:
            predictions = brain.solve_task(task, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)

            # Check correctness
            task_correct = False
            for j, gt in enumerate(task['test_outputs']):
                gt_array = np.array(gt, dtype=np.uint8)
                total += 1

                for attempt in predictions[j]:
                    pred = np.array(attempt, dtype=np.uint8)
                    if np.array_equal(pred, gt_array):
                        correct += 1
                        task_correct = True
                        break

            results.append({
                'task_id': task['id'],
                'time': elapsed,
                'correct': task_correct
            })

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            total += len(task.get('test_outputs', []))
            results.append({
                'task_id': task['id'],
                'time': elapsed,
                'correct': False,
                'error': str(e)
            })

        if (i + 1) % 10 == 0:
            avg_time = np.mean(times)
            acc = (correct / total * 100) if total > 0 else 0
            stats = brain.get_brain_stats()
            print(f"  Progress: {i+1}/{len(tasks)} | "
                  f"Avg: {avg_time*1000:.1f}ms | "
                  f"Acc: {acc:.2f}% | "
                  f"Brain age: {stats['age_seconds']:.1f}s")

    # Final results
    avg_time = np.mean(times)
    accuracy = (correct / total * 100) if total > 0 else 0

    stats = brain.get_brain_stats()

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Tasks processed: {len(tasks)}")
    print(f"Avg time per task: {avg_time*1000:.2f}ms")
    print(f"Total time: {sum(times):.3f}s")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"\nBrain statistics:")
    print(f"  Total tasks in lifetime: {stats['tasks_processed']}")
    print(f"  Brain age: {stats['age_seconds']:.2f}s")
    print(f"  Memory size: {stats['memory_size']}")
    print(f"  Status: {'ALIVE' if stats['alive'] else 'DEAD'}")
    print("="*80)

    # Save results
    output = {
        'num_tasks': len(tasks),
        'avg_time_ms': avg_time * 1000,
        'total_time': sum(times),
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'brain_stats': stats,
        'results': results
    }

    timestamp = int(time.time())
    output_file = f"results/living_brain_benchmark_{num_tasks}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("\nThe brain continues living, ready for more tasks...")

if __name__ == "__main__":
    main()

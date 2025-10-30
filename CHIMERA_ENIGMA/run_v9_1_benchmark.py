#!/usr/bin/env python3
"""
Benchmark CHIMERA v9.1 - Spatial Transformation Brain

Goal: Achieve 5%+ accuracy with persistent GPU architecture and spatial reasoning.
"""

import json
import time
import numpy as np
import sys

from chimera_v9_1 import get_brain_v9_1

def load_arc_tasks(num_tasks=100):
    """Load official ARC-AGI2 training tasks"""
    # This path needs to be correct relative to where you run the script
    challenges_file = "data/arc-agi_training_challenges.json"
    solutions_file = "data/arc-agi_training_solutions.json"

    try:
        with open(challenges_file, 'r') as f:
            challenges = json.load(f)

        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
    except FileNotFoundError:
        print("Error: ARC data files not found.")
        print("Please ensure 'data/arc-agi_training_challenges.json' and 'data/arc-agi_training_solutions.json' exist.")
        sys.exit(1)

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
    print("CHIMERA v9.1 - BENCHMARK")
    print("="*80)
    print("Spatial transformation brain with persistent GPU architecture")
    print("Target: 5%+ accuracy")
    print("="*80)

    # Load tasks
    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks from official training set")

    # Get brain
    print("\nAwakening v9.1 brain...")
    brain = get_brain_v9_1()
    print(f"Brain ready: {brain.ctx.info['GL_RENDERER']}\n")

    # Benchmark
    times = []
    correct = 0
    total_pairs = 0
    tasks_solved = 0
    results = []

    for i, task in enumerate(tasks):
        start = time.time()

        try:
            predictions = brain.solve_task(task, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)

            # Check correctness
            is_task_correct = True
            if not task['test_outputs']:
                is_task_correct = False
            
            for j, test_case_solutions in enumerate(task['test_outputs']):
                total_pairs += 1
                # ARC can have multiple valid solutions, check against any
                any_solution_match = False
                for attempt in predictions[j]:
                    pred_array = np.array(attempt, dtype=np.uint8)
                    for solution in test_case_solutions:
                        gt_array = np.array(solution, dtype=np.uint8)
                        if np.array_equal(pred_array, gt_array):
                            any_solution_match = True
                            break
                    if any_solution_match:
                        break
                
                if not any_solution_match:
                    is_task_correct = False

            if is_task_correct:
                tasks_solved += 1

            results.append({
                'task_id': task['id'],
                'time': elapsed,
                'correct': is_task_correct
            })

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            results.append({
                'task_id': task['id'],
                'time': elapsed,
                'correct': False,
                'error': str(e)
            })

        if (i + 1) % 10 == 0:
            avg_time = np.mean(times)
            acc = (tasks_solved / (i + 1) * 100) if (i + 1) > 0 else 0
            stats = brain.get_stats()
            print(f"  Progress: {i+1}/{len(tasks)} | "
                  f"Avg: {avg_time*1000:.2f}ms | "
                  f"Acc: {acc:.2f}% ({tasks_solved}/{i+1}) | "
                  f"Brain age: {stats['age_seconds']:.1f}s")

    # Final results
    avg_time = np.mean(times)
    accuracy = (tasks_solved / len(tasks) * 100) if tasks else 0
    stats = brain.get_stats()

    print("\n" + "="*80)
    print("CHIMERA v9.1 - RESULTS")
    print("="*80)
    print(f"Tasks processed: {len(tasks)}")
    print(f"Avg time per task: {avg_time*1000:.2f}ms")
    print(f"Total time: {sum(times):.3f}s")
    print(f"Accuracy: {accuracy:.2f}% ({tasks_solved}/{len(tasks)})")
    print(f"\nBrain statistics:")
    print(f"  Version: {stats['version']}")
    print(f"  Total tasks: {stats['tasks_processed']}")
    print(f"  Brain age: {stats['age_seconds']:.2f}s")
    print(f"  Status: {'ALIVE' if stats['alive'] else 'DEAD'}")
    print("="*80)

    # Compare to target
    if accuracy >= 20.0:
        print(f"\nSUCCESS! Target of 20% achieved! Current accuracy: {accuracy:.2f}%")
    else:
        print(f"\nTarget not reached. Gap: {20.0 - accuracy:.2f}%")
    print("="*80)

    # Save results
    output = {
        'version': '9.1',
        'num_tasks': len(tasks),
        'avg_time_ms': avg_time * 1000,
        'total_time': sum(times),
        'accuracy': accuracy,
        'tasks_solved': tasks_solved,
        'total_tasks': len(tasks),
        'brain_stats': stats,
        'results': results
    }

    timestamp = int(time.time())
    output_file = f"results/v9_1_benchmark_{num_tasks}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
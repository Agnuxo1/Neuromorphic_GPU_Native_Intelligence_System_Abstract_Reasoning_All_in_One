#!/usr/bin/env python3
"""
Benchmark CHIMERA v9.5 - Pattern Decoder

Neuromorphic evolution with temporal pattern recognition
"""

import json
import time
import numpy as np
import sys
from chimera_v9_5 import get_brain_v95

def load_arc_tasks(num_tasks=100):
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
            'test': task_data['test'],
            'test_outputs': []
        }
        if task_id in solutions:
            for sol in solutions[task_id]:
                task['test_outputs'].append(sol)
        tasks.append(task)
    return tasks

def main():
    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("="*80)
    print("CHIMERA v9.5 - PATTERN DECODER BENCHMARK")
    print("="*80)
    print("'No es AGI, es criptografia' - Turing approach")
    print("Neuromorphic loop: Estado + Memoria + Resultado = 1 fotograma")
    print("="*80)

    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks\n")

    brain = get_brain_v95()
    print(f"Brain ready: {brain.ctx.info['GL_RENDERER']}\n")

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

            results.append({'task_id': task['id'], 'time': elapsed, 'correct': task_correct})
        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            total += len(task.get('test_outputs', []))
            results.append({'task_id': task['id'], 'time': elapsed, 'correct': False, 'error': str(e)})

        if (i + 1) % 10 == 0:
            avg_time = np.mean(times)
            acc = (correct / total * 100) if total > 0 else 0
            stats = brain.get_stats()
            print(f"  Progress: {i+1}/{len(tasks)} | Avg: {avg_time*1000:.2f}ms | Acc: {acc:.2f}% | Age: {stats['age_seconds']:.1f}s")

    avg_time = np.mean(times)
    accuracy = (correct / total * 100) if total > 0 else 0
    stats = brain.get_stats()

    print("\n" + "="*80)
    print("CHIMERA v9.5 - RESULTS")
    print("="*80)
    print(f"Tasks: {len(tasks)}")
    print(f"Avg time: {avg_time*1000:.2f}ms")
    print(f"Total time: {sum(times):.3f}s")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"\nBrain stats:")
    print(f"  Version: {stats['version']}")
    print(f"  Tasks: {stats['tasks_processed']}")
    print(f"  Age: {stats['age_seconds']:.2f}s")
    print(f"  Status: {'ALIVE' if stats['alive'] else 'DEAD'}")
    print("="*80)

    if accuracy >= 5.0:
        print("\nSUCCESS! 5%+ achieved with pattern decoding!")
    else:
        print(f"\nGap to 5%: {5.0 - accuracy:.2f}%")
    print("="*80)

    output = {
        'version': '9.5',
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
    output_file = f"results/v9_5_benchmark_{num_tasks}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

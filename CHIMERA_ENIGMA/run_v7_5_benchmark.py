#!/usr/bin/env python3
"""
Benchmark for CHIMERA v7.5 on official ARC-AGI2 training set
"""

import json
import sys
from pathlib import Path
from chimera_v7_5 import solve_arc_task
import numpy as np
import time

def load_arc_tasks(num_tasks=100):
    """Load official ARC-AGI2 training tasks with solutions"""
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
        print(f"  ERROR in {task['id']}: {e}")
        return {
            'task_id': task['id'],
            'correct': 0,
            'total': len(task.get('test_outputs', [])),
            'accuracy': 0,
            'time': elapsed,
            'status': 'error',
            'error': str(e)
        }

def run_benchmark(num_tasks=100, verbose=False):
    """Run benchmark on official ARC-AGI2 training set"""
    print("=" * 80)
    print("CHIMERA v7.5 - OFFICIAL ARC-AGI2 TRAINING SET BENCHMARK")
    print("=" * 80)
    print("Dataset: Official ARC-AGI2 Training (1000 tasks)")
    print("Improvements over v7.2 (4%):")
    print("  + PatternDetector (task classification)")
    print("  + ConditionalColorMapper (context-aware)")
    print("  + GridDiffer (difference analysis)")
    print("  + StrategyRouter (smart routing)")
    print("=" * 80)
    print(f"Target: 15-20% accuracy")
    print("=" * 80)

    # Load tasks
    tasks = load_arc_tasks(num_tasks)
    print(f"\nLoaded {len(tasks)} tasks from official training set")
    print(f"Evaluating first {num_tasks} tasks...\n")

    # Evaluate
    results = []
    total_correct = 0
    total_tests = 0
    total_time = 0

    for i, task in enumerate(tasks):
        if not verbose:
            print(f"Task {i+1}/{num_tasks}: {task['id']}", end=" ... ")

        result = evaluate_task(task, verbose=verbose)
        results.append(result)

        total_correct += result['correct']
        total_tests += result['total']
        total_time += result['time']

        if not verbose:
            status = "OK" if result['correct'] > 0 else "FAIL"
            print(f"{status} {result['accuracy']:.0f}% ({result['correct']}/{result['total']}) {result['time']:.3f}s")

        # Progress update every 20 tasks
        if (i + 1) % 20 == 0:
            current_acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
            print(f"\n{'='*80}")
            print(f"Progress: {i+1}/{num_tasks} tasks")
            print(f"Cumulative Accuracy: {current_acc:.2f}%")
            print(f"Avg time: {total_time/(i+1):.3f}s/task")
            print(f"{'='*80}\n")

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    avg_time = total_time / len(results) if results else 0

    print(f"Dataset: Official ARC-AGI2 Training Set")
    print(f"Tasks evaluated: {len(results)}")
    print(f"Test cases: {total_tests}")
    print(f"Correct predictions: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"Average time per task: {avg_time:.3f}s")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 80)

    # Compare with v7.2
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS VERSIONS")
    print("=" * 80)
    print(f"v7.2 (baseline): 1.92% accuracy, 0.023s/task (on 100 tasks)")
    print(f"v7.5 (this run): {overall_accuracy:.2f}% accuracy, {avg_time:.3f}s/task")

    if overall_accuracy > 1.92:
        improvement = overall_accuracy - 1.92
        improvement_factor = overall_accuracy / 1.92 if 1.92 > 0 else 0
        print(f"\nImprovement: +{improvement:.2f} pp ({improvement_factor:.1f}x better) ✓✓✓")
    else:
        print(f"\nNo improvement (regression)")

    print("=" * 80)

    # Context
    print("\n" + "=" * 80)
    print("CONTEXT: STATE OF THE ART ON ARC-AGI2")
    print("=" * 80)
    print("OpenAI o3 (low): 4% ($200/task)")
    print("GPT-4.5, Claude, Gemini: ~1%")
    print("DeepSeek R1, o1-pro: 1-1.3%")
    print(f"CHIMERA v7.5: {overall_accuracy:.2f}% ($0/task)")
    print("=" * 80)

    # Save results
    output_file = f"results/benchmark_v7_5_{int(time.time())}.json"
    Path("results").mkdir(exist_ok=True)

    summary = {
        'version': 'v7.5',
        'dataset': 'official_arc_agi2_training',
        'improvements': ['PatternDetector', 'ConditionalColorMapper', 'GridDiffer', 'StrategyRouter'],
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

    print(f"\nDetailed results saved to: {output_file}\n")

    return summary

if __name__ == "__main__":
    num_tasks = 100
    verbose = False

    if len(sys.argv) > 1:
        num_tasks = int(sys.argv[1])

    if len(sys.argv) > 2 and sys.argv[2] == '-v':
        verbose = True

    run_benchmark(num_tasks=num_tasks, verbose=verbose)

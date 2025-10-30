#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script for CHIMERA v9.6
Tests on ARC-AGI-2 training tasks
"""

import json
import sys
import time
from pathlib import Path

# Import v9.6 brain
from chimera_v9_6 import get_brain_v96, to_list, to_grid


def load_tasks(dataset_path="../data/arc-agi_training_challenges.json"):
    """Load ARC-AGI training tasks."""
    try:
        with open(dataset_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: Cannot find {dataset_path}")
        return {}


def evaluate_predictions(predictions, expected):
    """
    Check if any of the 3 candidates matches the expected output.
    Returns True if any match.
    """
    for candidate in predictions:
        if candidate == expected:
            return True
    return False


def run_benchmark(num_tasks=100):
    """Run benchmark on first num_tasks from ARC training set."""
    print("="*80)
    print("CHIMERA v9.6 - BENCHMARK")
    print("="*80)

    # Load tasks
    tasks_dict = load_tasks()
    if not tasks_dict:
        print("No tasks loaded. Exiting.")
        return

    task_ids = list(tasks_dict.keys())[:num_tasks]
    print(f"Testing on {len(task_ids)} tasks")
    print("="*80)

    # Get living brain
    brain = get_brain_v96()

    # Stats
    correct = 0
    total_test_items = 0
    times = []

    start_all = time.time()

    for idx, task_id in enumerate(task_ids):
        task = tasks_dict[task_id]

        start = time.time()
        try:
            predictions = brain.solve_task(task, verbose=False)
        except Exception as e:
            print(f"[{idx+1}/{len(task_ids)}] {task_id}: ERROR - {e}")
            times.append(0)
            total_test_items += len(task.get("test", []))
            continue
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

        # Check accuracy
        task_correct = 0
        for ti, test_item in enumerate(task.get("test", [])):
            if ti < len(predictions):
                expected = test_item.get("output", [])
                if evaluate_predictions(predictions[ti], expected):
                    correct += 1
                    task_correct += 1
            total_test_items += 1

        status = "OK" if task_correct > 0 else "MISS"
        print(f"[{idx+1}/{len(task_ids)}] {task_id}: {elapsed:.1f}ms | {task_correct}/{len(task.get('test', []))} | {status}")

    elapsed_all = time.time() - start_all

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    accuracy = (correct / total_test_items * 100) if total_test_items > 0 else 0
    avg_time = sum(times) / len(times) if times else 0

    print(f"Tasks processed: {len(task_ids)}")
    print(f"Total test items: {total_test_items}")
    print(f"Correct predictions: {correct}/{total_test_items}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg time per task: {avg_time:.2f}ms")
    print(f"Total time: {elapsed_all:.2f}s")
    print(f"Throughput: {len(task_ids)/elapsed_all:.2f} tasks/sec")
    print("="*80)

    # Brain stats
    print("\nBrain stats:", brain.get_stats())


if __name__ == "__main__":
    num = 100
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    run_benchmark(num)

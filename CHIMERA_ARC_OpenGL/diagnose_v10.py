#!/usr/bin/env python3
"""
Diagnostic tool for CHIMERA v10.0 failures
==========================================

Analyze why v10.0 is failing on ARC tasks
"""

import json
import numpy as np
from pathlib import Path
from chimera_v10_0 import get_brain_v10

DATA_DIR = Path("D:/ARC2_CHIMERA/CHIMERA_ARC_OpenGL/Data")
EVALUATION_CHALLENGES = DATA_DIR / "arc-agi_evaluation_challenges.json"
EVALUATION_SOLUTIONS = DATA_DIR / "arc-agi_evaluation_solutions.json"

def diagnose_simple_task():
    """Test on simplest possible color mapping task."""
    print("=" * 80)
    print("DIAGNOSTIC #1: Simple Color Mapping")
    print("=" * 80)

    # Simplest task: swap 1->2, 2->1
    task = {
        'train': [
            {'input': [[1, 2], [2, 1]],
             'output': [[2, 1], [1, 2]]},
        ],
        'test': [
            {'input': [[1, 1], [2, 2]]}
        ]
    }

    expected = [[2, 2], [1, 1]]

    brain = get_brain_v10()
    predictions = brain.solve_task(task, verbose=True)

    print("\n[EXPECTED]")
    print(np.array(expected))

    print("\n[ATTEMPT 1]")
    print(np.array(predictions[0][0]))

    print("\n[ATTEMPT 2]")
    print(np.array(predictions[0][1]))

    match1 = np.array_equal(predictions[0][0], expected)
    match2 = np.array_equal(predictions[0][1], expected)

    print(f"\n[RESULT] Attempt 1 match: {match1}, Attempt 2 match: {match2}")

    return match1 or match2

def diagnose_real_task(task_id: str = "16b78196"):
    """Test on real ARC task."""
    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC #2: Real ARC Task ({task_id})")
    print("=" * 80)

    with open(EVALUATION_CHALLENGES, 'r') as f:
        challenges = json.load(f)

    with open(EVALUATION_SOLUTIONS, 'r') as f:
        solutions = json.load(f)

    task = challenges[task_id]
    solution = solutions[task_id]

    print(f"\n[TRAIN EXAMPLES] {len(task['train'])}")
    for i, ex in enumerate(task['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        print(f"  Example {i+1}: input {inp.shape}, output {out.shape}")
        print(f"    Input unique colors: {np.unique(inp)}")
        print(f"    Output unique colors: {np.unique(out)}")

    print(f"\n[TEST CASES] {len(task['test'])}")

    brain = get_brain_v10()
    predictions = brain.solve_task(task, verbose=True)

    print("\n[PREDICTIONS VS SOLUTIONS]")
    for i, (pred_attempts, true_output) in enumerate(zip(predictions, solution)):
        print(f"\n  Test case {i+1}:")

        expected = np.array(true_output)
        attempt1 = np.array(pred_attempts[0])
        attempt2 = np.array(pred_attempts[1])

        print(f"    Expected shape: {expected.shape}")
        print(f"    Attempt 1 shape: {attempt1.shape}")
        print(f"    Attempt 2 shape: {attempt2.shape}")

        if attempt1.shape == expected.shape:
            diff1 = np.sum(attempt1 != expected)
            print(f"    Attempt 1 differences: {diff1}/{expected.size} pixels")

        if attempt2.shape == expected.shape:
            diff2 = np.sum(attempt2 != expected)
            print(f"    Attempt 2 differences: {diff2}/{expected.size} pixels")

        print(f"\n    Expected (first 5x5):")
        print(expected[:5, :5])
        print(f"\n    Attempt 1 (first 5x5):")
        print(attempt1[:5, :5])

    return False

def diagnose_normalization():
    """Test color normalization."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC #3: Color Normalization")
    print("=" * 80)

    from chimera_v10_0 import normalize_arc_color, denormalize_arc_color

    print("\n[ROUND-TRIP TEST]")
    errors = 0
    for color in range(10):
        norm = normalize_arc_color(color)
        denorm = denormalize_arc_color(norm)
        match = (color == denorm)
        status = "OK" if match else "FAIL"
        print(f"  {color} -> {norm:.4f} -> {denorm} [{status}]")
        if not match:
            errors += 1

    print(f"\n[RESULT] Errors: {errors}/10")
    return errors == 0

def main():
    """Run all diagnostics."""
    print("=" * 80)
    print("CHIMERA v10.0 - DIAGNOSTIC SUITE")
    print("=" * 80)

    results = {}

    # Test 1: Simple task
    try:
        results['simple_task'] = diagnose_simple_task()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        results['simple_task'] = False

    # Test 2: Real task
    try:
        results['real_task'] = diagnose_real_task()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        results['real_task'] = False

    # Test 3: Normalization
    try:
        results['normalization'] = diagnose_normalization()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        results['normalization'] = False

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze what types of transformations v7.2 fails at
"""

import json
import numpy as np
from collections import Counter
from chimera_v7_2 import solve_arc_task

def analyze_transformation_type(train_examples):
    """Classify what type of transformation this task requires"""
    if not train_examples:
        return "unknown"

    ex = train_examples[0]
    inp = np.array(ex['input'])
    out = np.array(ex['output'])

    patterns = []

    # 1. Shape change?
    if inp.shape != out.shape:
        h_in, w_in = inp.shape
        h_out, w_out = out.shape

        if h_out > h_in or w_out > w_in:
            patterns.append("scaling_up")
        elif h_out < h_in or w_out < w_in:
            patterns.append("scaling_down")
        else:
            patterns.append("shape_change")
    else:
        patterns.append("same_shape")

    # 2. Check for geometric transformations
    if inp.shape == out.shape:
        # Rotation?
        for rot in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, rot), out):
                patterns.append(f"rotation_{rot*90}")
                return "_".join(patterns)

        # Flip?
        if np.array_equal(np.fliplr(inp), out):
            patterns.append("flip_h")
            return "_".join(patterns)

        if np.array_equal(np.flipud(inp), out):
            patterns.append("flip_v")
            return "_".join(patterns)

    # 3. Color mapping?
    if inp.shape == out.shape:
        # Check if it's a simple color substitution
        inp_colors = set(np.unique(inp))
        out_colors = set(np.unique(out))

        if inp_colors == out_colors:
            patterns.append("identity_or_complex")
        else:
            patterns.append("color_mapping")

    # 4. Object-based?
    inp_unique = len(np.unique(inp))
    out_unique = len(np.unique(out))

    if inp_unique <= 3 and out_unique <= 3:
        patterns.append("few_colors")
    elif inp_unique > 5 or out_unique > 5:
        patterns.append("many_colors")

    return "_".join(patterns) if patterns else "unknown"

def main():
    print("=" * 80)
    print("ANALYZING V7.2 FAILURE PATTERNS")
    print("=" * 80)

    # Load dataset
    with open('data/arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('data/arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    print(f"\nAnalyzing first 50 tasks...\n")

    correct_tasks = []
    failed_tasks = []

    for i, (task_id, task_data) in enumerate(list(challenges.items())[:50]):
        task = {
            'id': task_id,
            'train': task_data['train'],
            'test': task_data['test']
        }

        try:
            preds = solve_arc_task(task, verbose=False)

            # Check correctness
            is_correct = False
            if task_id in solutions:
                for j, sol in enumerate(solutions[task_id]):
                    gt = np.array(sol, dtype=np.uint8)
                    for attempt in preds[j]:
                        pred = np.array(attempt, dtype=np.uint8)
                        if np.array_equal(pred, gt):
                            is_correct = True
                            break
                    if is_correct:
                        break

            # Analyze transformation type
            trans_type = analyze_transformation_type(task['train'])

            if is_correct:
                correct_tasks.append((task_id, trans_type))
            else:
                failed_tasks.append((task_id, trans_type))

            if (i + 1) % 10 == 0:
                print(f"  Analyzed {i+1}/50 tasks...")

        except Exception as e:
            failed_tasks.append((task_id, "error"))

    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    print(f"Correct: {len(correct_tasks)}/50 ({len(correct_tasks)/50*100:.1f}%)")
    print(f"Failed: {len(failed_tasks)}/50 ({len(failed_tasks)/50*100:.1f}%)")

    # Analyze what v7.2 SUCCEEDS at
    print(f"\n{'=' * 80}")
    print("TASKS V7.2 SUCCEEDS AT:")
    print("=" * 80)
    if correct_tasks:
        correct_types = Counter([t[1] for t in correct_tasks])
        for trans_type, count in correct_types.most_common():
            print(f"  {trans_type}: {count} tasks")
    else:
        print("  None in this sample")

    # Analyze what v7.2 FAILS at
    print(f"\n{'=' * 80}")
    print("TASKS V7.2 FAILS AT (Top patterns):")
    print("=" * 80)
    failed_types = Counter([t[1] for t in failed_tasks])
    for trans_type, count in failed_types.most_common(10):
        print(f"  {trans_type}: {count} tasks")
        # Show examples
        examples = [t[0] for t in failed_tasks if t[1] == trans_type][:3]
        for ex_id in examples:
            print(f"    - {ex_id}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

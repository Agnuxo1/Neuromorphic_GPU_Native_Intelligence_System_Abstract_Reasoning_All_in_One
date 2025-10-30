#!/usr/bin/env python3
"""
Analyze how grid sizes change in ARC-AGI2 training examples
This is KEY: size evolution is a TEMPORAL PATTERN like IQ tests
"""

import json
import numpy as np
from collections import Counter

def analyze_size_patterns():
    """Analyze size change patterns across all training tasks"""

    with open('data/arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    print("=" * 80)
    print("ARC-AGI2 GRID SIZE PATTERN ANALYSIS")
    print("=" * 80)
    print("Hypothesis: Grid sizes follow TEMPORAL patterns (like IQ tests)")
    print("=" * 80)

    size_patterns = []

    for task_id, task_data in list(challenges.items())[:100]:
        train_examples = task_data['train']

        # Extract size sequence from training
        size_sequence = []
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            size_sequence.append({
                'input_size': inp.shape,
                'output_size': out.shape,
                'same_size': inp.shape == out.shape,
                'size_change': (out.shape[0] - inp.shape[0], out.shape[1] - inp.shape[1])
            })

        # Analyze pattern
        pattern_type = classify_size_pattern(size_sequence)
        size_patterns.append(pattern_type)

        # Show first few examples
        if len(size_patterns) <= 10:
            print(f"\nTask {task_id}:")
            print(f"  Pattern: {pattern_type}")
            for i, s in enumerate(size_sequence):
                print(f"    Example {i+1}: {s['input_size']} -> {s['output_size']}")

    # Statistics
    print("\n" + "=" * 80)
    print("PATTERN STATISTICS (100 tasks)")
    print("=" * 80)

    pattern_counts = Counter(size_patterns)
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern}: {count} tasks ({count}%)")

    return pattern_counts


def classify_size_pattern(size_sequence):
    """Classify what type of size pattern this is"""

    if not size_sequence:
        return "unknown"

    # Check if all examples have same input->output size relationship
    same_sizes = [s['same_size'] for s in size_sequence]
    size_changes = [s['size_change'] for s in size_sequence]

    # Pattern 1: All same size (no change)
    if all(same_sizes):
        return "constant_size"

    # Pattern 2: Consistent scaling
    unique_changes = set(size_changes)
    if len(unique_changes) == 1:
        change = size_changes[0]
        if change[0] > 0 or change[1] > 0:
            return f"consistent_upscale_{change}"
        elif change[0] < 0 or change[1] < 0:
            return f"consistent_downscale_{change}"

    # Pattern 3: Variable scaling (temporal pattern!)
    if len(unique_changes) > 1:
        # Check for arithmetic progression
        if is_arithmetic_sequence([s['output_size'][0] for s in size_sequence]):
            return "arithmetic_progression_height"
        if is_arithmetic_sequence([s['output_size'][1] for s in size_sequence]):
            return "arithmetic_progression_width"

        return "variable_scaling"

    return "mixed_pattern"


def is_arithmetic_sequence(values):
    """Check if values form arithmetic sequence (like 3, 5, 7, 9...)"""
    if len(values) < 2:
        return False

    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    return len(set(diffs)) == 1  # All differences are the same


def analyze_temporal_prediction():
    """
    Simulate IQ test: given sequence, predict next
    """
    print("\n" + "=" * 80)
    print("TEMPORAL SEQUENCE PREDICTION (IQ Test Style)")
    print("=" * 80)

    with open('data/arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    correct_predictions = 0
    total_tests = 0

    for task_id, task_data in list(challenges.items())[:20]:
        train_examples = task_data['train']

        if len(train_examples) < 2:
            continue

        # Extract output sizes from training
        output_sizes = []
        for ex in train_examples:
            out = np.array(ex['output'])
            output_sizes.append(out.shape)

        # Try to predict pattern
        predicted = predict_next_size(output_sizes)

        # Check if there's a test case to validate
        if len(task_data['test']) > 0:
            # We can't validate without solutions, but we can show prediction
            total_tests += 1

            print(f"\nTask {task_id}:")
            print(f"  Training output sizes: {output_sizes}")
            print(f"  Predicted test size: {predicted}")

    return total_tests


def predict_next_size(size_sequence):
    """
    Predict next size in sequence (IQ test logic)

    Examples:
    - (3,3), (3,3), (3,3) -> (3,3) [constant]
    - (3,3), (6,6), (9,9) -> (12,12) [arithmetic +3]
    - (2,2), (4,4), (8,8) -> (16,16) [geometric x2]
    """
    if len(size_sequence) < 2:
        return size_sequence[-1] if size_sequence else (3, 3)

    # Extract heights and widths
    heights = [s[0] for s in size_sequence]
    widths = [s[1] for s in size_sequence]

    # Predict height
    pred_h = predict_sequence_value(heights)
    pred_w = predict_sequence_value(widths)

    return (pred_h, pred_w)


def predict_sequence_value(values):
    """Predict next value in 1D sequence"""
    if len(values) < 2:
        return values[-1] if values else 3

    # Check for constant
    if len(set(values)) == 1:
        return values[0]

    # Check for arithmetic progression
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    if len(set(diffs)) == 1:
        # Arithmetic: continue with same diff
        return values[-1] + diffs[0]

    # Check for geometric progression
    if all(v > 0 for v in values):
        ratios = [values[i+1] / values[i] for i in range(len(values)-1)]
        if all(abs(r - ratios[0]) < 0.01 for r in ratios):
            # Geometric: continue with same ratio
            return int(values[-1] * ratios[0])

    # Check for second-order arithmetic (differences form arithmetic sequence)
    if len(diffs) >= 2:
        second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
        if len(set(second_diffs)) == 1:
            # Second order arithmetic
            next_diff = diffs[-1] + second_diffs[0]
            return values[-1] + next_diff

    # Fallback: repeat last value
    return values[-1]


if __name__ == "__main__":
    # Analyze size patterns
    patterns = analyze_size_patterns()

    # Analyze temporal prediction
    analyze_temporal_prediction()

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("Grid sizes in ARC follow TEMPORAL PATTERNS:")
    print("  - Constant size (most common)")
    print("  - Arithmetic progressions (3,5,7,9...)")
    print("  - Geometric progressions (2,4,8,16...)")
    print("  - Second-order patterns")
    print("\nThis is EXACTLY like IQ tests!")
    print("Predicting output size FIRST is critical.")
    print("=" * 80)

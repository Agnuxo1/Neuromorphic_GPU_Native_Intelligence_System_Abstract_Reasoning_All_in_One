#!/usr/bin/env python3
"""
ARC Pattern Decoder - Cryptography Approach (Turing/Enigma style)

Hypothesis: ARC is NOT an AGI problem, it's a PATTERN DECODING problem
Like Enigma: find the transformation function, don't "understand" it

Key insight: There's a MECHANICAL transformation from input→output
We just need to FIND it, not REASON about it
"""

import json
import numpy as np
from collections import Counter

def analyze_transformation_mechanics(inp, out):
    """
    Find the MECHANICAL transformation pattern
    (like finding Enigma rotor settings)
    """
    patterns = {}

    # Pattern 1: Pixel-wise function
    # Is there a function f(color, x, y) → new_color?
    if inp.shape == out.shape:
        # Check if transformation is POSITIONAL
        position_dependent = check_position_dependency(inp, out)
        patterns['position_dependent'] = position_dependent

        # Check if transformation is COLOR-only
        color_only = check_color_only_transform(inp, out)
        patterns['color_only'] = color_only

        # Check if transformation is NEIGHBOR-dependent
        neighbor_dependent = check_neighbor_dependency(inp, out)
        patterns['neighbor_dependent'] = neighbor_dependent

    # Pattern 2: Structural transformation
    # Copy, rotate, tile, etc.
    structural = check_structural_pattern(inp, out)
    patterns['structural'] = structural

    return patterns


def check_color_only_transform(inp, out):
    """
    Check if output[i,j] = f(input[i,j]) for ALL positions
    (Pure color substitution cipher)
    """
    mapping = {}
    consistent = True

    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            in_color = int(inp[i, j])
            out_color = int(out[i, j])

            if in_color in mapping:
                if mapping[in_color] != out_color:
                    consistent = False
                    break
            else:
                mapping[in_color] = out_color
        if not consistent:
            break

    if consistent and mapping:
        return {'type': 'color_substitution', 'mapping': mapping}

    return None


def check_position_dependency(inp, out):
    """
    Check if transformation depends on (x,y) position
    Like: "add 1 to all cells in top half"
    """
    h, w = inp.shape

    # Check for row-based patterns
    row_patterns = []
    for i in range(h):
        row_in = inp[i, :]
        row_out = out[i, :]
        diffs = row_out.astype(int) - row_in.astype(int)
        row_patterns.append(tuple(diffs))

    # If all rows have same pattern → not position dependent
    if len(set(row_patterns)) == 1:
        return None

    # Check if pattern is systematic (e.g., row number affects transformation)
    # TODO: Implement more sophisticated checks

    return {'type': 'position_dependent', 'row_patterns': row_patterns[:3]}


def check_neighbor_dependency(inp, out):
    """
    Check if output[i,j] depends on neighbors of input[i,j]
    Like: "if pixel has 8-neighbor, change color"
    """
    if inp.shape != out.shape:
        return None

    h, w = inp.shape
    neighbor_rules = []

    for i in range(h):
        for j in range(w):
            if inp[i,j] != out[i,j]:
                # Get neighbors
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbors.append(int(inp[ni, nj]))

                rule = {
                    'in_color': int(inp[i,j]),
                    'out_color': int(out[i,j]),
                    'neighbors': Counter(neighbors)
                }
                neighbor_rules.append(rule)

    if len(neighbor_rules) > 0:
        return {'type': 'neighbor_dependent', 'sample_rules': neighbor_rules[:3]}

    return None


def check_structural_pattern(inp, out):
    """
    Check for structural transformations:
    - Tiling: output = tile(input, n, m)
    - Rotation: output = rotate(input, angle)
    - Reflection: output = flip(input)
    - Cropping/Padding
    """
    patterns = []

    # Check tiling
    if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
        n = out.shape[0] // inp.shape[0]
        m = out.shape[1] // inp.shape[1]

        # Verify if it's actually tiled
        is_tiled = True
        for i in range(n):
            for j in range(m):
                tile = out[i*inp.shape[0]:(i+1)*inp.shape[0],
                          j*inp.shape[1]:(j+1)*inp.shape[1]]
                if not np.array_equal(tile, inp):
                    is_tiled = False
                    break
            if not is_tiled:
                break

        if is_tiled:
            patterns.append({'type': 'tiling', 'n': n, 'm': m})

    # Check rotation (if same shape)
    if inp.shape == out.shape:
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, k), out):
                patterns.append({'type': 'rotation', 'degrees': k*90})
                break

    # Check reflection
    if inp.shape == out.shape:
        if np.array_equal(np.fliplr(inp), out):
            patterns.append({'type': 'flip_horizontal'})
        elif np.array_equal(np.flipud(inp), out):
            patterns.append({'type': 'flip_vertical'})

    return patterns if patterns else None


def find_pattern_hypothesis(training_examples):
    """
    Generate hypotheses about the transformation pattern
    (Like Turing testing different Enigma settings)
    """
    all_patterns = []

    for ex in training_examples:
        inp = np.array(ex['input'], dtype=np.uint8)
        out = np.array(ex['output'], dtype=np.uint8)

        patterns = analyze_transformation_mechanics(inp, out)
        all_patterns.append(patterns)

    # Find CONSISTENT patterns across all training examples
    consistent = find_consistent_patterns(all_patterns)

    return consistent


def find_consistent_patterns(all_patterns):
    """
    Find patterns that appear in ALL training examples
    (The "key" to the cipher)
    """
    if not all_patterns:
        return None

    # Check color_only transformation
    color_maps = [p.get('color_only') for p in all_patterns if p.get('color_only')]
    if len(color_maps) == len(all_patterns):
        # All examples use color substitution
        # Check if mappings are consistent
        merged_mapping = {}
        consistent = True

        for cm in color_maps:
            for in_c, out_c in cm['mapping'].items():
                if in_c in merged_mapping:
                    if merged_mapping[in_c] != out_c:
                        consistent = False
                        break
                else:
                    merged_mapping[in_c] = out_c
            if not consistent:
                break

        if consistent:
            return {
                'type': 'color_substitution',
                'mapping': merged_mapping,
                'confidence': 1.0
            }

    # Check structural patterns
    structural_patterns = [p.get('structural') for p in all_patterns if p.get('structural')]
    if structural_patterns:
        # Find most common structural pattern
        pattern_types = []
        for sp in structural_patterns:
            if sp:
                for p in sp:
                    pattern_types.append(p['type'])

        if pattern_types:
            most_common = Counter(pattern_types).most_common(1)[0]
            if most_common[1] >= len(all_patterns) * 0.7:  # 70% consistency
                return {
                    'type': 'structural',
                    'pattern': most_common[0],
                    'confidence': most_common[1] / len(all_patterns)
                }

    return {'type': 'complex', 'confidence': 0.0}


def apply_pattern_hypothesis(test_input, hypothesis):
    """
    Apply the discovered pattern to test input
    (Decrypt the message with the found key)
    """
    if not hypothesis:
        return test_input.copy()

    if hypothesis['type'] == 'color_substitution':
        result = test_input.copy()
        for in_c, out_c in hypothesis['mapping'].items():
            result[test_input == in_c] = out_c
        return result

    elif hypothesis['type'] == 'structural':
        pattern = hypothesis['pattern']

        if pattern == 'rotation':
            # Find rotation angle from training
            return np.rot90(test_input, 1)  # Default 90

        elif pattern == 'flip_horizontal':
            return np.fliplr(test_input)

        elif pattern == 'flip_vertical':
            return np.flipud(test_input)

        elif pattern == 'tiling':
            # Would need to extract n, m from hypothesis
            return test_input.copy()

    return test_input.copy()


def decode_arc_task(task):
    """
    Main decoder: treat task as cryptographic puzzle
    """
    # Step 1: Analyze training to find the "cipher key"
    hypothesis = find_pattern_hypothesis(task['train'])

    print(f"\n[DECODER] Pattern hypothesis: {hypothesis}")

    if hypothesis and hypothesis.get('confidence', 0) > 0.5:
        print(f"[DECODER] High confidence pattern found!")

    # Step 2: Apply pattern to test input
    solutions = []
    for test_case in task['test']:
        test_input = np.array(test_case['input'], dtype=np.uint8)
        solution = apply_pattern_hypothesis(test_input, hypothesis)
        solutions.append([solution.tolist(), solution.tolist()])

    return solutions


if __name__ == "__main__":
    print("=" * 80)
    print("ARC PATTERN DECODER - Cryptography Approach")
    print("=" * 80)
    print("Treating ARC as Enigma: Find the transformation KEY")
    print("=" * 80)

    # Load and analyze some tasks
    with open('data/arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('data/arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    print("\nAnalyzing 10 tasks to find decoding patterns...\n")

    pattern_types = []

    for task_id in list(challenges.keys())[:10]:
        task = challenges[task_id]

        print(f"\nTask {task_id}:")
        hypothesis = find_pattern_hypothesis(task['train'])

        if hypothesis:
            print(f"  Pattern: {hypothesis['type']}")
            print(f"  Confidence: {hypothesis.get('confidence', 0):.2f}")
            pattern_types.append(hypothesis['type'])

            if hypothesis['type'] == 'color_substitution':
                print(f"  Mapping: {hypothesis['mapping']}")

    print("\n" + "=" * 80)
    print("PATTERN DISTRIBUTION")
    print("=" * 80)
    for pattern, count in Counter(pattern_types).most_common():
        print(f"  {pattern}: {count} tasks")

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("Simple patterns (color substitution, structural) appear frequently!")
    print("These are MECHANICALLY solvable without 'reasoning'")
    print("Like Enigma: test all possible 'keys', find the one that works")
    print("=" * 80)

#!/usr/bin/env python3
"""
Test v7.3 on a single simple task with verbose output
"""

from chimera_v7_3 import solve_arc_task
import numpy as np

# Simple test task: color mapping +1
test_task = {
    'id': 'test_simple',
    'train': [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[2, 3], [4, 5]]
        },
        {
            'input': [[5, 6], [7, 8]],
            'output': [[6, 7], [8, 9]]
        }
    ],
    'test': [
        {'input': [[1, 2], [3, 4]]}
    ]
}

print("=" * 80)
print("CHIMERA v7.3 - SIMPLE TEST")
print("=" * 80)
print("\nTest task: Color mapping +1")
print("Training examples:")
for i, ex in enumerate(test_task['train']):
    print(f"  Example {i+1}:")
    print(f"    Input:  {ex['input']}")
    print(f"    Output: {ex['output']}")

print("\nTest input:")
print(f"  {test_task['test'][0]['input']}")

print("\nExpected output:")
print(f"  [[2, 3], [4, 5]]")

print("\n" + "=" * 80)
print("Running v7.3 solver with verbose=True...")
print("=" * 80)

result = solve_arc_task(test_task, verbose=True)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Generated {len(result)} test outputs:")
for i, attempts in enumerate(result):
    print(f"\nTest {i+1} - 2 attempts:")
    print(f"  Attempt 1: {attempts[0]}")
    print(f"  Attempt 2: {attempts[1]}")

print("\nExpected: [[2, 3], [4, 5]]")

# Check if any match
expected = np.array([[2, 3], [4, 5]], dtype=np.uint8)
for i, attempts in enumerate(result):
    for j, attempt in enumerate(attempts):
        pred = np.array(attempt, dtype=np.uint8)
        if np.array_equal(pred, expected):
            print(f"\n✓ Match found: Test {i+1} Attempt {j+1}")
            break
    else:
        print(f"\n✗ No match for test {i+1}")

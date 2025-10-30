#!/usr/bin/env python3
"""
CHIMERA v10.0 - Comprehensive Test Suite

Tests all major components:
1. Color normalization
2. Frame operations
3. Spatial operators
4. Object extraction
5. DSL operators
6. Beam search
7. Hungarian algorithm
8. Dual attempts
9. Full task solving

Run: python test_chimera_v10.py
"""

import numpy as np
import time
from chimera_v10_0 import (
    LivingBrainV10,
    NeuromorphicFrameV10,
    ObjectExtractor,
    CHIMERA_DSL,
    BeamSearchSolver,
    normalize_arc_color,
    denormalize_arc_color,
    hungarian_color_mapping,
    solve_arc_task
)


class Colors:
    """ANSI color codes for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_failure(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")


def print_section(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")


def test_color_normalization():
    """Test 1: Color normalization and denormalization."""
    print_section("TEST 1: Color Normalization")
    
    test_cases = [
        (0, 0.0),      # Background
        (1, 0.2),      # First object color
        (5, 0.6),      # Middle color
        (9, 1.0),      # Last color
    ]
    
    all_passed = True
    
    for color, expected_norm in test_cases:
        norm = normalize_arc_color(color, background_aware=True)
        denorm = denormalize_arc_color(norm)
        
        norm_ok = abs(norm - expected_norm) < 0.01
        denorm_ok = denorm == color
        
        if norm_ok and denorm_ok:
            print_success(f"Color {color}: {norm:.3f} -> {denorm}")
        else:
            print_failure(f"Color {color}: expected {expected_norm:.3f}, got {norm:.3f}")
            all_passed = False
    
    # Test roundtrip for all colors
    for color in range(10):
        norm = normalize_arc_color(color)
        denorm = denormalize_arc_color(norm)
        if denorm != color:
            print_failure(f"Roundtrip failed for color {color}")
            all_passed = False
    
    if all_passed:
        print_success("All color normalization tests passed")
    
    return all_passed


def test_frame_operations():
    """Test 2: Frame creation and operations."""
    print_section("TEST 2: Frame Operations")
    
    brain = LivingBrainV10()
    
    # Test frame creation
    frame = NeuromorphicFrameV10(brain.ctx, (5, 5))
    print_success("Frame created successfully")
    
    # Test state upload
    test_grid = np.array([
        [0, 1, 2, 3, 4],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]
    ], dtype=np.uint8)
    
    frame.upload_state(test_grid)
    print_success("State uploaded to GPU")
    
    # Test download
    result = frame.download_result()
    
    # Result should initially be close to input (before evolution)
    # B channel starts as zeros, so result will be all 0s initially
    if result.shape == (5, 5):
        print_success("State downloaded from GPU")
    else:
        print_failure(f"Shape mismatch: expected (5,5), got {result.shape}")
        return False
    
    # Test position encoding
    if frame.position_texture is not None:
        print_success("Position encoding texture created")
    
    frame.release()
    print_success("Frame resources released")
    
    return True


def test_spatial_operators():
    """Test 3: 3×3 spatial operators."""
    print_section("TEST 3: Spatial Operators")
    
    brain = LivingBrainV10()
    
    # Create cross pattern
    cross = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)
    
    frame = NeuromorphicFrameV10(brain.ctx, (5, 5))
    frame.upload_state(cross)
    
    # Evolve with identity mapping
    identity_map = list(range(10))
    frame = brain._neuromorphic_evolution(frame, identity_map, steps=1)
    
    # Download spatial features
    features = frame.download_spatial_features()
    
    # Center pixel (2, 2) should have:
    # - High density (surrounded by same color)
    # - Not edge (interior of shape)
    center_density = features[2, 2, 1]  # G channel
    
    # Edge pixels should be detected
    edge_pixel = features[0, 2, 0]  # R channel (edge_strength)
    
    tests_passed = 0
    
    if center_density > 0.5:
        print_success(f"Center density: {center_density:.2f} (high as expected)")
        tests_passed += 1
    else:
        print_failure(f"Center density: {center_density:.2f} (expected high)")
    
    if edge_pixel > 0.5:
        print_success(f"Edge detection: {edge_pixel:.2f} (detected)")
        tests_passed += 1
    else:
        print_failure(f"Edge detection: {edge_pixel:.2f} (not detected)")
    
    frame.release()
    
    return tests_passed == 2


def test_object_extraction():
    """Test 4: Connected components extraction."""
    print_section("TEST 4: Object Extraction")
    
    brain = LivingBrainV10()
    extractor = ObjectExtractor(brain.ctx)
    
    # Create grid with 3 separate objects
    grid = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 3, 0, 0],
        [3, 3, 3, 0, 0]
    ], dtype=np.uint8)
    
    frame = NeuromorphicFrameV10(brain.ctx, (5, 5))
    frame.upload_state(grid)
    
    # Extract components
    labels_tex = extractor.compute_connected_components(frame)
    
    # Read labels
    rgba = np.frombuffer(labels_tex.read(), dtype=np.float32)
    rgba = rgba.reshape((5, 5, 4))
    
    labels = rgba[:, :, 1]  # G channel = component id
    
    # Check that same objects have same label
    obj1_labels = labels[0:2, 0:2].flatten()  # Top-left object
    obj2_labels = labels[0:2, 3:5].flatten()  # Top-right object
    
    obj1_same = len(np.unique(obj1_labels)) == 1
    obj2_same = len(np.unique(obj2_labels)) == 1
    obj_different = obj1_labels[0] != obj2_labels[0]
    
    if obj1_same and obj2_same and obj_different:
        print_success("Objects correctly labeled as separate components")
        labels_tex.release()
        frame.release()
        return True
    else:
        print_failure("Object labeling failed")
        labels_tex.release()
        frame.release()
        return False


def test_dsl_operators():
    """Test 5: DSL geometric operators."""
    print_section("TEST 5: DSL Operators")
    
    brain = LivingBrainV10()
    dsl = CHIMERA_DSL(brain)
    
    # Test grid
    grid = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.uint8)
    
    frame = NeuromorphicFrameV10(brain.ctx, (2, 2))
    frame.upload_state(grid)
    
    tests_passed = 0
    
    # Test rotate_90
    op = next(op for op in dsl.operators if op.name == 'rotate_90')
    rotated_frame = dsl.apply_operator(frame, op)
    result = rotated_frame.download_result()
    
    expected_90 = np.array([[3, 1], [4, 2]])
    
    if np.array_equal(result, expected_90):
        print_success("rotate_90: Correct")
        tests_passed += 1
    else:
        print_failure(f"rotate_90: Expected\n{expected_90}\nGot\n{result}")
    
    rotated_frame.release()
    
    # Test flip_h
    frame2 = NeuromorphicFrameV10(brain.ctx, (2, 2))
    frame2.upload_state(grid)
    
    op_flip = next(op for op in dsl.operators if op.name == 'flip_h')
    flipped_frame = dsl.apply_operator(frame2, op_flip)
    result_flip = flipped_frame.download_result()
    
    expected_flip = np.array([[2, 1], [4, 3]])
    
    if np.array_equal(result_flip, expected_flip):
        print_success("flip_h: Correct")
        tests_passed += 1
    else:
        print_failure(f"flip_h: Expected\n{expected_flip}\nGot\n{result_flip}")
    
    flipped_frame.release()
    frame2.release()
    frame.release()
    
    return tests_passed >= 2


def test_hungarian_algorithm():
    """Test 6: Hungarian color mapping."""
    print_section("TEST 6: Hungarian Algorithm")
    
    # Create training examples with clear mapping: 1->2, 2->3
    train = [
        {
            'input': [[1, 1, 2, 2]],
            'output': [[2, 2, 3, 3]]
        },
        {
            'input': [[2, 1, 1, 2]],
            'output': [[3, 2, 2, 3]]
        }
    ]
    
    color_map = hungarian_color_mapping(train)
    
    # Check mapping
    if color_map[1] == 2 and color_map[2] == 3:
        print_success(f"Hungarian mapping correct: 1→{color_map[1]}, 2→{color_map[2]}")
        return True
    else:
        print_failure(f"Hungarian mapping incorrect: {color_map}")
        return False


def test_beam_search():
    """Test 7: Beam search solver."""
    print_section("TEST 7: Beam Search")
    
    brain = LivingBrainV10()
    dsl = CHIMERA_DSL(brain)
    searcher = BeamSearchSolver(dsl, beam_width=4, max_depth=2)
    
    # Simple task: rotate 90
    train = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[3, 1], [4, 2]]
        }
    ]
    
    test_input = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    
    print("Running beam search (this may take a few seconds)...")
    start = time.time()
    result = searcher.search(train, test_input)
    elapsed = time.time() - start
    
    expected = np.array([[7, 5], [8, 6]])
    
    if np.array_equal(result, expected):
        print_success(f"Beam search found correct solution in {elapsed:.2f}s")
        return True
    else:
        print_failure(f"Beam search result incorrect:\nExpected:\n{expected}\nGot:\n{result}")
        return False


def test_dual_attempts():
    """Test 8: Dual attempt strategy."""
    print_section("TEST 8: Dual Attempt Strategy")
    
    task = {
        'train': [
            {'input': [[0, 1, 1], [1, 1, 0]], 
             'output': [[0, 2, 2], [2, 2, 0]]},
            {'input': [[1, 0, 1], [0, 1, 1]], 
             'output': [[2, 0, 2], [0, 2, 2]]},
        ],
        'test': [
            {'input': [[1, 1, 1], [1, 1, 1]]}
        ]
    }
    
    brain = LivingBrainV10()
    result = brain.solve_task(task, verbose=False)
    
    # Should return 2 attempts
    if len(result) == 1 and len(result[0]) == 2:
        attempt1 = np.array(result[0][0])
        attempt2 = np.array(result[0][1])
        
        print_success(f"Dual attempts generated: {attempt1.shape} and {attempt2.shape}")
        
        # At least one should have mostly 2s (the mapped color)
        if np.sum(attempt1 == 2) > 3 or np.sum(attempt2 == 2) > 3:
            print_success("At least one attempt has correct color mapping")
            return True
        else:
            print_failure("Neither attempt has correct color mapping")
            return False
    else:
        print_failure(f"Expected 2 attempts, got {len(result[0]) if result else 0}")
        return False


def test_full_task_solving():
    """Test 9: Full task solving pipeline."""
    print_section("TEST 9: Full Task Solving")
    
    # Test multiple task types
    test_tasks = [
        {
            'name': 'Color Mapping',
            'task': {
                'train': [
                    {'input': [[1, 2], [2, 1]], 'output': [[3, 4], [4, 3]]},
                ],
                'test': [{'input': [[2, 2], [1, 1]]}]
            },
            'expected_contains': [3, 4]  # Should contain these colors
        },
        {
            'name': 'Size Change',
            'task': {
                'train': [
                    {'input': [[1, 1]], 'output': [[1, 1], [1, 1]]},
                ],
                'test': [{'input': [[2, 2]]}]
            },
            'expected_shape': (2, 2)
        },
        {
            'name': 'Identity',
            'task': {
                'train': [
                    {'input': [[1, 2, 3]], 'output': [[1, 2, 3]]},
                ],
                'test': [{'input': [[4, 5, 6]]}]
            },
            'expected_shape': (1, 3)
        }
    ]
    
    brain = LivingBrainV10()
    all_passed = True
    
    for test_case in test_tasks:
        print(f"\n  Testing: {test_case['name']}")
        
        try:
            result = brain.solve_task(test_case['task'], verbose=False)
            attempt1 = np.array(result[0][0])
            
            # Check shape if specified
            if 'expected_shape' in test_case:
                if attempt1.shape == test_case['expected_shape']:
                    print_success(f"  Shape correct: {attempt1.shape}")
                else:
                    print_failure(f"  Shape wrong: expected {test_case['expected_shape']}, got {attempt1.shape}")
                    all_passed = False
            
            # Check contains colors if specified
            if 'expected_contains' in test_case:
                unique_colors = np.unique(attempt1).tolist()
                if any(c in unique_colors for c in test_case['expected_contains']):
                    print_success(f"  Contains expected colors")
                else:
                    print_failure(f"  Missing expected colors")
                    all_passed = False
        
        except Exception as e:
            print_failure(f"  Exception: {str(e)}")
            all_passed = False
    
    return all_passed


def test_performance():
    """Test 10: Performance benchmarking."""
    print_section("TEST 10: Performance Benchmark")
    
    task = {
        'train': [
            {'input': [[1, 2], [3, 4]], 'output': [[2, 3], [4, 5]]},
        ],
        'test': [{'input': [[5, 6], [7, 8]]}]
    }
    
    brain = LivingBrainV10()
    
    # Warmup
    brain.solve_task(task, verbose=False)
    
    # Benchmark
    num_runs = 10
    times = []
    
    print(f"Running {num_runs} iterations...")
    for i in range(num_runs):
        start = time.time()
        brain.solve_task(task, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    
    print_success(f"Average time: {avg_time:.2f}ms ± {std_time:.2f}ms")
    
    if avg_time < 500:  # Less than 500ms is good
        print_success("Performance: Excellent (< 500ms)")
        return True
    elif avg_time < 1000:
        print_success("Performance: Good (< 1s)")
        return True
    else:
        print_failure(f"Performance: Slow (> 1s)")
        return False


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}CHIMERA v10.0 - COMPREHENSIVE TEST SUITE{Colors.END}")
    print("="*60)
    
    tests = [
        ("Color Normalization", test_color_normalization),
        ("Frame Operations", test_frame_operations),
        ("Spatial Operators", test_spatial_operators),
        ("Object Extraction", test_object_extraction),
        ("DSL Operators", test_dsl_operators),
        ("Hungarian Algorithm", test_hungarian_algorithm),
        ("Beam Search", test_beam_search),
        ("Dual Attempts", test_dual_attempts),
        ("Full Task Solving", test_full_task_solving),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print_failure(f"Exception in {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {status}  {name}")
    
    print(f"\n{Colors.BOLD}Results: {passed_count}/{total_count} tests passed{Colors.END}")
    
    if passed_count == total_count:
        print(f"{Colors.GREEN}{Colors.BOLD}")
        print("="*60)
        print("  ✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("  CHIMERA v10.0 is ready for ARC-AGI 2025!")
        print("="*60)
        print(Colors.END)
        return True
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}")
        print("="*60)
        print(f"  {passed_count}/{total_count} tests passed")
        print("  Some components need attention")
        print("="*60)
        print(Colors.END)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
CHIMERA v7.6 - Output Size Prediction First (IQ Test Approach)

KEY INSIGHT: ARC grid sizes follow TEMPORAL PATTERNS like IQ tests
- Constant: (3,3), (3,3), (3,3) → (3,3)
- Arithmetic: (3,3), (6,6), (9,9) → (12,12)
- Geometric: (2,2), (4,4), (8,8) → (16,16)

V7.6 Strategy:
1. FIRST: Predict output size from training sequence (IQ test logic)
2. THEN: Generate content for that specific size
3. VALIDATE: Ensure output matches predicted size

This is the CRITICAL missing piece in v7.2 and v7.5!

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import numpy as np
import moderngl
from typing import Tuple, Optional, List, Dict
from collections import Counter
import time


class OutputSizePredictor:
    """
    Predict output grid size based on temporal sequence (IQ test logic)

    This is KEY: treat size prediction as a separate IQ-test problem
    """

    def predict_output_size(self, training_examples: List[Dict], test_input_shape: Tuple) -> Tuple:
        """
        Predict output size for test input based on training sequence

        Like IQ test: 3, 5, 7, 9, ? → answer: 11
        """
        if not training_examples:
            return test_input_shape

        # Extract sequences
        input_sizes = []
        output_sizes = []

        for ex in training_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            input_sizes.append(inp.shape)
            output_sizes.append(out.shape)

        # Strategy 1: Check for constant output size (75% of cases)
        if self._is_constant_size(output_sizes):
            # Return most common size
            return Counter(output_sizes).most_common(1)[0][0]

        # Strategy 2: Check for consistent scaling ratio
        scaling_ratio = self._detect_scaling_ratio(input_sizes, output_sizes)
        if scaling_ratio:
            h_scale, w_scale = scaling_ratio
            pred_h = int(test_input_shape[0] * h_scale)
            pred_w = int(test_input_shape[1] * w_scale)
            return self._clamp_size((pred_h, pred_w))

        # Strategy 3: Temporal sequence prediction (IQ test)
        pred_h = self._predict_sequence([s[0] for s in output_sizes])
        pred_w = self._predict_sequence([s[1] for s in output_sizes])

        return self._clamp_size((pred_h, pred_w))

    def _is_constant_size(self, sizes: List[Tuple]) -> bool:
        """Check if all sizes are the same"""
        return len(set(sizes)) == 1

    def _detect_scaling_ratio(self, input_sizes: List[Tuple], output_sizes: List[Tuple]) -> Optional[Tuple]:
        """Detect if there's a consistent scaling ratio"""
        ratios = []

        for inp_size, out_size in zip(input_sizes, output_sizes):
            if inp_size[0] > 0 and inp_size[1] > 0:
                h_ratio = out_size[0] / inp_size[0]
                w_ratio = out_size[1] / inp_size[1]
                ratios.append((h_ratio, w_ratio))

        if not ratios:
            return None

        # Check if all ratios are the same
        if len(set(ratios)) == 1:
            ratio = ratios[0]
            # Only return if it's a clean integer ratio
            if ratio[0] == int(ratio[0]) and ratio[1] == int(ratio[1]):
                return (int(ratio[0]), int(ratio[1]))

        return None

    def _predict_sequence(self, values: List[int]) -> int:
        """
        Predict next value in 1D sequence (IQ test logic)

        Examples:
        - 3, 3, 3 → 3 (constant)
        - 3, 5, 7 → 9 (arithmetic +2)
        - 2, 4, 8 → 16 (geometric ×2)
        - 1, 2, 4, 7 → 11 (second-order: differences are 1,2,3 → next is 4)
        """
        if len(values) == 0:
            return 3  # Default

        if len(values) == 1:
            return values[0]

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
            ratios = [values[i+1] / values[i] for i in range(len(values)-1) if values[i] > 0]
            if ratios and all(abs(r - ratios[0]) < 0.01 for r in ratios):
                # Geometric: continue with same ratio
                return int(values[-1] * ratios[0])

        # Check for second-order arithmetic
        if len(diffs) >= 2:
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            if len(set(second_diffs)) == 1:
                # Second order: differences form arithmetic sequence
                next_diff = diffs[-1] + second_diffs[0]
                return values[-1] + next_diff

        # Fallback: repeat last value
        return values[-1]

    def _clamp_size(self, size: Tuple) -> Tuple:
        """Clamp size to ARC limits [1, 30]"""
        h = max(1, min(30, size[0]))
        w = max(1, min(30, size[1]))
        return (h, w)


class ContentGenerator:
    """
    Generate grid content for a specific target size

    This takes the predicted size and generates appropriate content
    """

    def generate_content(self, test_input: np.ndarray, target_size: Tuple,
                        training_examples: List[Dict]) -> np.ndarray:
        """
        Generate content that fits target_size

        Strategies:
        1. If same size as input → transform in-place
        2. If larger → upscale intelligently
        3. If smaller → downscale intelligently
        """
        test_h, test_w = test_input.shape
        target_h, target_w = target_size

        # Strategy 1: Same size
        if (test_h, test_w) == (target_h, target_w):
            return self._transform_same_size(test_input, training_examples)

        # Strategy 2: Upscale
        elif target_h >= test_h and target_w >= test_w:
            return self._upscale(test_input, target_size, training_examples)

        # Strategy 3: Downscale
        elif target_h <= test_h and target_w <= test_w:
            return self._downscale(test_input, target_size, training_examples)

        # Strategy 4: Mixed (one dimension up, one down)
        else:
            return self._mixed_scale(test_input, target_size, training_examples)

    def _transform_same_size(self, test_input: np.ndarray, training: List[Dict]) -> np.ndarray:
        """Transform input of same size (v7.2 approach)"""
        # Learn color mapping
        color_map = {}

        for ex in training:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)

            if inp.shape == out.shape:
                for color in np.unique(inp):
                    mask = (inp == color)
                    out_colors = out[mask]
                    if len(out_colors) > 0:
                        most_common = Counter(out_colors).most_common(1)[0][0]
                        if color not in color_map:
                            color_map[color] = most_common

        # Apply mapping
        result = test_input.copy()
        for old_color, new_color in color_map.items():
            result[test_input == old_color] = new_color

        return result

    def _upscale(self, test_input: np.ndarray, target_size: Tuple, training: List[Dict]) -> np.ndarray:
        """Upscale to larger size"""
        target_h, target_w = target_size
        test_h, test_w = test_input.shape

        # Calculate scaling factors
        h_scale = target_h // test_h if test_h > 0 else 1
        w_scale = target_w // test_w if test_w > 0 else 1

        # Use Kronecker product for integer scaling
        if h_scale > 0 and w_scale > 0:
            scaled = np.kron(test_input, np.ones((h_scale, w_scale), dtype=np.uint8))

            # Crop or pad to exact target size
            if scaled.shape[0] > target_h:
                scaled = scaled[:target_h, :]
            if scaled.shape[1] > target_w:
                scaled = scaled[:, :target_w]

            if scaled.shape[0] < target_h or scaled.shape[1] < target_w:
                # Pad with zeros
                padded = np.zeros(target_size, dtype=np.uint8)
                padded[:scaled.shape[0], :scaled.shape[1]] = scaled
                return padded

            return scaled

        # Fallback: create blank grid
        return np.zeros(target_size, dtype=np.uint8)

    def _downscale(self, test_input: np.ndarray, target_size: Tuple, training: List[Dict]) -> np.ndarray:
        """Downscale to smaller size"""
        target_h, target_w = target_size
        test_h, test_w = test_input.shape

        # Simple downsampling
        h_step = test_h // target_h if target_h > 0 else 1
        w_step = test_w // target_w if target_w > 0 else 1

        return test_input[::h_step, ::w_step][:target_h, :target_w]

    def _mixed_scale(self, test_input: np.ndarray, target_size: Tuple, training: List[Dict]) -> np.ndarray:
        """Handle mixed scaling (one dimension up, one down)"""
        # For now, create blank grid of target size
        # TODO: Implement smarter mixed scaling
        return np.zeros(target_size, dtype=np.uint8)


class UnifiedReasoningEngineV76:
    """
    CHIMERA v7.6 - Output Size Prediction First

    Critical improvement: Predict output size BEFORE generating content
    Treats size prediction as IQ test (temporal sequence)
    """

    def __init__(self, use_gpu: bool = True, verbose: bool = False):
        self.use_gpu = use_gpu
        self.verbose = verbose

        print("=" * 80)
        print("CHIMERA v7.6 - OUTPUT SIZE PREDICTION FIRST (IQ TEST APPROACH)")
        print("=" * 80)
        print("Key: Predict size from temporal sequence, THEN generate content")
        print("=" * 80)

        # Initialize GPU context
        self.ctx = None
        if use_gpu:
            try:
                self.ctx = moderngl.create_standalone_context()
                if self.verbose:
                    print(f"[GPU] {self.ctx.info['GL_RENDERER']}")
            except Exception as e:
                if self.verbose:
                    print(f"[GPU] Not available: {e}")
                self.use_gpu = False

        # Initialize components
        self.size_predictor = OutputSizePredictor()
        self.content_generator = ContentGenerator()

    def solve_task(self, task: Dict) -> List[List]:
        """Solve with size prediction first"""

        if self.verbose:
            print(f"\n[TASK] Training: {len(task['train'])}, Test: {len(task['test'])}")

        solutions = []

        for test_idx, test_case in enumerate(task['test']):
            test_input = np.array(test_case['input'], dtype=np.uint8)

            if self.verbose:
                print(f"\n[TEST {test_idx}] Input shape: {test_input.shape}")

            # STEP 1: Predict output size (IQ test logic)
            predicted_size = self.size_predictor.predict_output_size(
                task['train'],
                test_input.shape
            )

            if self.verbose:
                print(f"[SIZE PREDICTOR] Predicted output: {predicted_size}")

            # STEP 2: Generate content for that size
            solution = self.content_generator.generate_content(
                test_input,
                predicted_size,
                task['train']
            )

            if self.verbose:
                print(f"[CONTENT] Generated shape: {solution.shape}")

            # Validate size
            if solution.shape != predicted_size:
                if self.verbose:
                    print(f"[WARNING] Size mismatch! Adjusting...")
                # Adjust to match predicted size
                solution = self._adjust_to_size(solution, predicted_size)

            # Return two attempts
            # Attempt 1: Size-aware generation
            # Attempt 2: Same (for now)
            solutions.append([solution.tolist(), solution.tolist()])

        return solutions

    def _adjust_to_size(self, grid: np.ndarray, target_size: Tuple) -> np.ndarray:
        """Adjust grid to match target size"""
        target_h, target_w = target_size

        if grid.shape == target_size:
            return grid

        # Create blank grid of target size
        result = np.zeros(target_size, dtype=np.uint8)

        # Copy what fits
        copy_h = min(grid.shape[0], target_h)
        copy_w = min(grid.shape[1], target_w)

        result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

        return result

    def release(self):
        """Clean up"""
        if self.ctx:
            self.ctx.release()


def solve_arc_task(task: Dict, verbose: bool = False) -> List[List]:
    """Main entry point"""
    engine = UnifiedReasoningEngineV76(use_gpu=True, verbose=verbose)
    result = engine.solve_task(task)
    engine.release()
    return result


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHIMERA v7.6 - LOCAL TEST")
    print("=" * 80)

    # Test 1: Constant size
    test_task_1 = {
        'id': 'test_constant',
        'train': [
            {'input': [[1, 2], [3, 4]], 'output': [[2, 3], [4, 5]]},
            {'input': [[5, 6], [7, 8]], 'output': [[6, 7], [8, 9]]}
        ],
        'test': [{'input': [[1, 2], [3, 4]]}]
    }

    print("\nTest 1: Constant size (2,2) -> (2,2)")
    result = solve_arc_task(test_task_1, verbose=True)
    print(f"Result: {result[0][0]}")
    print(f"Expected: [[2, 3], [4, 5]]")

    # Test 2: Upscaling
    test_task_2 = {
        'id': 'test_upscale',
        'train': [
            {'input': [[1, 2]], 'output': [[1, 1, 2, 2]]},
            {'input': [[3, 4]], 'output': [[3, 3, 4, 4]]}
        ],
        'test': [{'input': [[5, 6]]}]
    }

    print("\n\nTest 2: Upscaling (1,2) -> (1,4)")
    result = solve_arc_task(test_task_2, verbose=True)
    print(f"Result: {result[0][0]}")

    print("\n" + "=" * 80)
    print("CHIMERA v7.6 READY FOR BENCHMARK")
    print("=" * 80)

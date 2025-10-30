#!/usr/bin/env python3
"""
CHIMERA v7.5 - Pattern Detection + Conditional Reasoning

Built on v7.2 (4% accuracy) with strategic improvements:

Phase 1 Enhancements:
1. PatternDetector: Classify task type (color mapping, spatial, object-based)
2. ConditionalColorMapper: Context-aware color mappings (not just global)
3. GridDiffer: Pixel-level difference analysis
4. StrategyRouter: Choose best strategy per task type

Target: 15-20% accuracy (4x improvement over v7.2)

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import numpy as np
import moderngl
from typing import Tuple, Optional, List, Dict, Set
from collections import Counter
from scipy import ndimage
import time


class PatternDetector:
    """
    Classify what type of transformation a task requires
    """

    def classify_task(self, training_examples: List[Dict]) -> str:
        """
        Analyze training examples and classify task type
        """
        if not training_examples:
            return "unknown"

        patterns = []
        for ex in training_examples:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)
            pattern = self._analyze_single_pair(inp, out)
            patterns.append(pattern)

        # Find most common pattern
        pattern_counts = Counter(patterns)
        most_common = pattern_counts.most_common(1)[0][0]

        return most_common

    def _analyze_single_pair(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Analyze a single input-output pair"""

        # 1. Check for shape change
        if inp.shape != out.shape:
            h_in, w_in = inp.shape
            h_out, w_out = out.shape

            if h_out > h_in or w_out > w_in:
                return "scaling_up"
            elif h_out < h_in or w_out < w_in:
                return "scaling_down"
            else:
                return "shape_change"

        # 2. Same shape - check transformation type
        # Check for geometric transformations
        for rot in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, rot), out):
                return "rotation"

        if np.array_equal(np.fliplr(inp), out):
            return "flip"

        if np.array_equal(np.flipud(inp), out):
            return "flip"

        # 3. Check if it's identity
        if np.array_equal(inp, out):
            return "identity"

        # 4. Count changes
        diff_ratio = np.sum(inp != out) / inp.size

        if diff_ratio < 0.1:
            return "sparse_changes"  # Few pixels change
        elif diff_ratio < 0.5:
            return "conditional_mapping"  # Moderate changes, likely conditional
        else:
            return "global_mapping"  # Most pixels change, likely global rule

    def needs_object_detection(self, training_examples: List[Dict]) -> bool:
        """Check if task requires object-level reasoning"""

        for ex in training_examples:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)

            # If there are distinct colored regions, likely object-based
            inp_colors = len(np.unique(inp))
            if inp_colors >= 3 and inp_colors <= 6:
                # Check for connected components
                if self._has_distinct_objects(inp):
                    return True

        return False

    def _has_distinct_objects(self, grid: np.ndarray) -> bool:
        """Check if grid has distinct objects (connected components)"""
        try:
            for color in np.unique(grid):
                if color == 0:  # Skip background
                    continue

                mask = (grid == color)
                labeled, num_features = ndimage.label(mask)

                if num_features >= 2:  # Multiple objects of same color
                    return True
        except:
            pass

        return False


class GridDiffer:
    """
    Analyze pixel-level differences between input and output
    """

    def analyze_differences(self, inp: np.ndarray, out: np.ndarray) -> Dict:
        """
        Find what changed and where
        """
        if inp.shape != out.shape:
            return {"type": "shape_mismatch"}

        diff_mask = (inp != out)
        num_changed = np.sum(diff_mask)

        if num_changed == 0:
            return {"type": "identity"}

        # Analyze change pattern
        changed_positions = np.argwhere(diff_mask)

        analysis = {
            "type": "changes",
            "num_changed": int(num_changed),
            "change_ratio": float(num_changed / inp.size),
            "changed_positions": changed_positions
        }

        # Analyze spatial pattern of changes
        analysis["spatial_pattern"] = self._analyze_spatial_pattern(changed_positions, inp.shape)

        # Analyze color changes
        analysis["color_changes"] = self._analyze_color_changes(inp, out, diff_mask)

        return analysis

    def _analyze_spatial_pattern(self, positions: np.ndarray, shape: Tuple) -> str:
        """Analyze if changes follow a spatial pattern"""
        if len(positions) == 0:
            return "none"

        h, w = shape

        # Check if changes are at edges
        edge_count = 0
        for y, x in positions:
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                edge_count += 1

        if edge_count / len(positions) > 0.8:
            return "edges"

        # Check if changes are clustered
        if len(positions) > 1:
            mean_y = np.mean(positions[:, 0])
            mean_x = np.mean(positions[:, 1])
            distances = np.sqrt((positions[:, 0] - mean_y) ** 2 + (positions[:, 1] - mean_x) ** 2)
            if np.mean(distances) < min(h, w) * 0.3:
                return "clustered"

        return "scattered"

    def _analyze_color_changes(self, inp: np.ndarray, out: np.ndarray, diff_mask: np.ndarray) -> Dict:
        """Analyze which colors change to which"""
        changes = {}

        changed_positions = np.argwhere(diff_mask)
        for y, x in changed_positions:
            from_color = int(inp[y, x])
            to_color = int(out[y, x])

            key = (from_color, to_color)
            if key not in changes:
                changes[key] = 0
            changes[key] += 1

        return changes


class ConditionalColorMapper:
    """
    Learn color mappings that depend on context (neighbors, position)
    """

    def __init__(self):
        self.rules = []

    def learn_from_training(self, training_examples: List[Dict]):
        """Extract conditional color mapping rules"""

        all_rules = []

        for ex in training_examples:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)

            if inp.shape != out.shape:
                continue

            rules = self._extract_rules(inp, out)
            all_rules.extend(rules)

        # Find consistent rules
        self.rules = self._find_consistent_rules(all_rules)

    def _extract_rules(self, inp: np.ndarray, out: np.ndarray) -> List[Dict]:
        """Extract rules from a single pair"""
        rules = []

        h, w = inp.shape
        for y in range(h):
            for x in range(w):
                inp_color = int(inp[y, x])
                out_color = int(out[y, x])

                if inp_color != out_color:
                    # Get context
                    neighbors = self._get_neighbors(inp, y, x)
                    position_type = self._classify_position(y, x, h, w)

                    rule = {
                        'from_color': inp_color,
                        'to_color': out_color,
                        'has_neighbor': neighbors,  # List of neighbor colors
                        'position': position_type
                    }
                    rules.append(rule)

        return rules

    def _get_neighbors(self, grid: np.ndarray, y: int, x: int) -> List[int]:
        """Get colors of neighboring cells"""
        h, w = grid.shape
        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbors.append(int(grid[ny, nx]))

        return neighbors

    def _classify_position(self, y: int, x: int, h: int, w: int) -> str:
        """Classify position type (edge, corner, center)"""
        if (y == 0 or y == h - 1) and (x == 0 or x == w - 1):
            return "corner"
        elif y == 0 or y == h - 1 or x == 0 or x == w - 1:
            return "edge"
        else:
            return "center"

    def _find_consistent_rules(self, all_rules: List[Dict]) -> List[Dict]:
        """Find rules that appear consistently"""
        # Simple heuristic: if a rule appears in majority of examples, keep it
        rule_counts = {}

        for rule in all_rules:
            # Create key based on transformation
            key = (rule['from_color'], rule['to_color'])

            if key not in rule_counts:
                rule_counts[key] = {'count': 0, 'rule': rule}
            rule_counts[key]['count'] += 1

        # Keep rules that appear at least once
        consistent = []
        for key, data in rule_counts.items():
            consistent.append(data['rule'])

        return consistent

    def apply_rules(self, test_input: np.ndarray) -> np.ndarray:
        """Apply learned conditional rules"""
        result = test_input.copy()

        if not self.rules:
            return result

        h, w = test_input.shape
        for y in range(h):
            for x in range(w):
                current_color = int(test_input[y, x])

                # Check each rule
                for rule in self.rules:
                    if rule['from_color'] == current_color:
                        # Check context
                        neighbors = self._get_neighbors(test_input, y, x)
                        position = self._classify_position(y, x, h, w)

                        # Simple matching: if position matches, apply
                        if rule['position'] == position:
                            result[y, x] = rule['to_color']
                            break  # Apply first matching rule

        return result


class StrategyRouter:
    """
    Route to appropriate solving strategy based on task type
    """

    def __init__(self, pattern_detector: PatternDetector):
        self.pattern_detector = pattern_detector

    def route(self, task: Dict) -> str:
        """Determine which strategy to use"""
        task_type = self.pattern_detector.classify_task(task['train'])
        needs_objects = self.pattern_detector.needs_object_detection(task['train'])

        # Priority routing
        if task_type == "rotation" or task_type == "flip":
            return "geometric"

        if task_type == "scaling_up" or task_type == "scaling_down":
            return "scaling"

        if task_type == "identity":
            return "identity"

        if needs_objects:
            return "object_based"

        if task_type == "conditional_mapping" or task_type == "sparse_changes":
            return "conditional"

        if task_type == "global_mapping":
            return "simple_mapping"

        return "fallback"


class UnifiedReasoningEngineV75:
    """
    CHIMERA v7.5 - Pattern Detection + Conditional Reasoning

    Improvements over v7.2:
    - Pattern detection to classify task type
    - Conditional color mapping (context-aware)
    - Grid diff analysis
    - Strategy routing
    """

    def __init__(self, use_gpu: bool = True, verbose: bool = False):
        self.use_gpu = use_gpu
        self.verbose = verbose

        print("=" * 80)
        print("CHIMERA v7.5 - PATTERN DETECTION + CONDITIONAL REASONING")
        print("=" * 80)
        print("Target: 15-20% accuracy (4x improvement over v7.2)")
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
        self.pattern_detector = PatternDetector()
        self.grid_differ = GridDiffer()
        self.conditional_mapper = ConditionalColorMapper()
        self.strategy_router = StrategyRouter(self.pattern_detector)

    def solve_task(self, task: Dict) -> List[List]:
        """Solve with pattern detection and conditional reasoning"""

        if self.verbose:
            print(f"\n[TASK] Training: {len(task['train'])}, Test: {len(task['test'])}")

        # Step 1: Detect task type
        task_type = self.pattern_detector.classify_task(task['train'])
        strategy = self.strategy_router.route(task)

        if self.verbose:
            print(f"[PATTERN] Task type: {task_type}")
            print(f"[STRATEGY] Selected: {strategy}")

        # Step 2: Learn from training
        self.conditional_mapper.learn_from_training(task['train'])

        # Step 3: Solve test cases
        solutions = []

        for test_idx, test_case in enumerate(task['test']):
            test_input = np.array(test_case['input'], dtype=np.uint8)

            if self.verbose:
                print(f"\n[TEST {test_idx}] Input shape: {test_input.shape}")

            # Route to strategy
            if strategy == "conditional":
                solution = self._solve_conditional(test_input, task['train'])
            elif strategy == "simple_mapping":
                solution = self._solve_simple_mapping(test_input, task['train'])
            elif strategy == "geometric":
                solution = self._solve_geometric(test_input, task['train'])
            elif strategy == "scaling":
                solution = self._solve_scaling(test_input, task['train'])
            elif strategy == "identity":
                solution = test_input.copy()
            else:
                solution = self._solve_fallback(test_input, task['train'])

            # Return two attempts
            # Attempt 1: Primary strategy
            # Attempt 2: Fallback (simple mapping)
            attempt2 = self._solve_simple_mapping(test_input, task['train'])

            solutions.append([solution.tolist(), attempt2.tolist()])

        return solutions

    def _solve_conditional(self, test_input: np.ndarray, training: List[Dict]) -> np.ndarray:
        """Solve with conditional color mapping"""
        return self.conditional_mapper.apply_rules(test_input)

    def _solve_simple_mapping(self, test_input: np.ndarray, training: List[Dict]) -> np.ndarray:
        """Solve with simple global color mapping (v7.2 style)"""
        # Learn simple mapping
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

    def _solve_geometric(self, test_input: np.ndarray, training: List[Dict]) -> np.ndarray:
        """Solve geometric transformations"""
        # Detect rotation/flip from training
        if not training:
            return test_input.copy()

        ex = training[0]
        inp = np.array(ex['input'], dtype=np.uint8)
        out = np.array(ex['output'], dtype=np.uint8)

        if inp.shape != out.shape:
            return test_input.copy()

        # Try rotations
        for rot in [1, 2, 3]:
            if np.array_equal(np.rot90(inp, rot), out):
                return np.rot90(test_input, rot)

        # Try flips
        if np.array_equal(np.fliplr(inp), out):
            return np.fliplr(test_input)

        if np.array_equal(np.flipud(inp), out):
            return np.flipud(test_input)

        return test_input.copy()

    def _solve_scaling(self, test_input: np.ndarray, training: List[Dict]) -> np.ndarray:
        """Solve scaling transformations"""
        if not training:
            return test_input.copy()

        # Infer scaling factor
        ex = training[0]
        inp = np.array(ex['input'], dtype=np.uint8)
        out = np.array(ex['output'], dtype=np.uint8)

        h_in, w_in = inp.shape
        h_out, w_out = out.shape

        if h_in > 0 and w_in > 0:
            h_scale = h_out / h_in
            w_scale = w_out / w_in

            if h_scale > 1:  # Upscaling
                h_s = int(h_scale)
                w_s = int(w_scale)
                if h_s == h_scale and w_s == w_scale:
                    return np.kron(test_input, np.ones((h_s, w_s), dtype=np.uint8))

            elif h_scale < 1:  # Downscaling
                h_s = int(1 / h_scale)
                w_s = int(1 / w_scale)
                return test_input[::h_s, ::w_s]

        return test_input.copy()

    def _solve_fallback(self, test_input: np.ndarray, training: List[Dict]) -> np.ndarray:
        """Fallback: try simple mapping"""
        return self._solve_simple_mapping(test_input, training)

    def release(self):
        """Clean up"""
        if self.ctx:
            self.ctx.release()


def solve_arc_task(task: Dict, verbose: bool = False) -> List[List]:
    """Main entry point"""
    engine = UnifiedReasoningEngineV75(use_gpu=True, verbose=verbose)
    result = engine.solve_task(task)
    engine.release()
    return result


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHIMERA v7.5 - LOCAL TEST")
    print("=" * 80)

    # Test: Color mapping
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

    result = solve_arc_task(test_task, verbose=True)
    print(f"\nGenerated solution:")
    print(f"  Attempt 1: {result[0][0]}")
    print(f"  Attempt 2: {result[0][1]}")
    print(f"\nExpected: [[2, 3], [4, 5]]")

    print("\n" + "=" * 80)
    print("CHIMERA v7.5 READY FOR BENCHMARK")
    print("=" * 80)

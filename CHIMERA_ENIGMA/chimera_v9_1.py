#!/usr/bin/env python3
"""
CHIMERA v9.1 - Living Brain with Spatial Transformation Solver

Extends v9.0 by adding a solver that tests for spatial transformations
(rotations and flips), significantly increasing its reasoning capabilities.

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import numpy as np
import moderngl
from typing import List, Dict, Tuple, Optional
import time
from collections import Counter


class LivingBrainV9_1:
    """
    Enhanced living brain with spatial transformation detection.
    """

    def __init__(self, memory_size=(256, 256)):
        """Birth of the enhanced brain"""
        print("="*80)
        print("CHIMERA v9.1 - SPATIAL TRANSFORMATION BRAIN")
        print("="*80)
        print("Persistent GPU architecture with spatial reasoning")
        print("="*80)

        # Create permanent GPU context
        self.ctx = moderngl.create_standalone_context()
        print(f"[GPU] {self.ctx.info['GL_RENDERER']}")

        # Persistent holographic memory
        self.memory_w, self.memory_h = memory_size
        self.holographic_memory = self.ctx.texture(
            size=(self.memory_w, self.memory_h),
            components=4,
            dtype='f4'
        )
        zeros = np.zeros((self.memory_h, self.memory_w, 4), dtype=np.float32)
        self.holographic_memory.write(zeros.tobytes())

        print(f"[MEMORY] Holographic memory: {self.memory_w}x{self.memory_h}")

        # Compile shaders
        self._compile_shaders()

        # Statistics
        self.tasks_processed = 0
        self.birth_time = time.time()
        self.successful_tasks = 0

        print(f"[BRAIN] v9.1 awakened and ready")
        print("="*80)

    def _compile_shaders(self):
        """Compile GPU shaders once"""
        vertex_shader = """
        #version 330
        in vec2 in_vert;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            uv = (in_vert + 1.0) / 2.0;
        }
        """

        # Enhanced color mapping shader
        fragment_shader_map = """
        #version 330
        uniform sampler2D u_input;
        uniform int u_color_map[10];
        uniform ivec2 grid_size;
        in vec2 uv;
        out vec4 out_color;

        void main() {
            ivec2 coord = ivec2(uv * grid_size);
            coord = clamp(coord, ivec2(0), grid_size - ivec2(1));

            vec4 input_pixel = texelFetch(u_input, coord, 0);
            int input_color = int(input_pixel.r * 9.0 + 0.5);
            input_color = clamp(input_color, 0, 9);

            int output_color = u_color_map[input_color];
            output_color = clamp(output_color, 0, 9);

            float color_val = float(output_color) / 9.0;
            out_color = vec4(color_val, 0.0, 0.0, 1.0);
        }
        """

        self.map_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader_map
        )

        print("[SHADERS] Enhanced shaders compiled")

    def _generate_spatial_transforms(self, grid: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates a dictionary of all 8 spatial transformations for a grid.
        """
        transforms = {
            'identity': grid,
            'rot90': np.rot90(grid, 1),
            'rot180': np.rot90(grid, 2),
            'rot270': np.rot90(grid, 3),
        }
        flipped_grid = np.fliplr(grid)
        transforms['flip'] = flipped_grid
        transforms['flip_rot90'] = np.rot90(flipped_grid, 1)
        transforms['flip_rot180'] = np.rot90(flipped_grid, 2)
        transforms['flip_rot270'] = np.rot90(flipped_grid, 3)
        return transforms

    def _predict_output_size(self, train_examples: List[Dict], test_input_shape: Tuple[int, int], transform_name: str = 'identity') -> Tuple[int, int]:
        """
        Predict output size based on training examples, considering transformations.
        """
        if not train_examples:
            return test_input_shape

        # Adjust test_input_shape based on transform
        if 'rot90' in transform_name or 'rot270' in transform_name:
            h, w = test_input_shape
            test_input_shape = (w, h)

        size_pairs = []
        for ex in train_examples:
            in_shape = np.array(ex['input']).shape
            out_shape = np.array(ex['output']).shape
            size_pairs.append((in_shape, out_shape))

        if len(set(s[1] for s in size_pairs)) == 1:
            return size_pairs[0][1]

        if all(s[0] == s[1] for s in size_pairs):
            return test_input_shape
        
        # More robust scaling logic needed here, for now, simple cases
        return test_input_shape


    def _learn_color_mapping(self, train_examples: List[Dict]) -> Tuple[List[int], float]:
        if not train_examples:
            return list(range(10)), 0.0
        mapping_counts = {}
        for example in train_examples:
            inp = np.array(example['input'], dtype=np.uint8)
            out = np.array(example['output'], dtype=np.uint8)
            if inp.shape != out.shape:
                continue
            for y in range(inp.shape[0]):
                for x in range(inp.shape[1]):
                    old_c, new_c = int(inp[y, x]), int(out[y, x])
                    if old_c not in mapping_counts:
                        mapping_counts[old_c] = Counter()
                    mapping_counts[old_c][new_c] += 1
        
        color_map = list(range(10))
        consistent_mappings = 0
        for old_c in range(10):
            if old_c in mapping_counts and mapping_counts[old_c]:
                new_c, count = mapping_counts[old_c].most_common(1)[0]
                color_map[old_c] = new_c
                if count / sum(mapping_counts[old_c].values()) > 0.8:
                    consistent_mappings += 1
        
        confidence = consistent_mappings / len(mapping_counts) if mapping_counts else 0.0
        return color_map, confidence

    def _apply_mapping_gpu(self, test_input: np.ndarray, color_map: List[int],
                          output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        h, w = test_input.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, 0] = test_input.astype(float) / 9.0
        rgba[:, :, 3] = 1.0
        input_tex = self.ctx.texture(size=(w, h), components=4, dtype='f4')
        input_tex.write(rgba.tobytes())

        out_h, out_w = output_size if output_size else (h, w)
        output_tex = self.ctx.texture(size=(out_w, out_h), components=4, dtype='f4')
        fbo = self.ctx.framebuffer(color_attachments=[output_tex])

        self.map_program['u_input'] = 0
        self.map_program['u_color_map'].write(np.array(color_map, dtype='i4').tobytes())
        self.map_program['grid_size'] = (w, h)
        input_tex.use(location=0)

        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._render_quad(self.map_program)

        rgba_out = np.frombuffer(output_tex.read(), dtype=np.float32).reshape((out_h, out_w, 4))
        result = (rgba_out[:, :, 0] * 9.0).round().astype(np.uint8)

        input_tex.release()
        output_tex.release()
        fbo.release()
        return result

    def solve_task(self, task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
        self.tasks_processed += 1
        if verbose:
            age = time.time() - self.birth_time
            print(f"\n[v9.1] Task #{self.tasks_processed} | Age: {age:.1f}s")

        start = time.time()

        # --- SPATIAL SOLVER ---
        best_transform_name = 'identity'
        best_transform_score = -1
        best_color_map = list(range(10))

        for transform_name, transform_func in self._generate_spatial_transforms(np.zeros((1,1))).items(): # Dummy grid
            is_consistent = True
            temp_color_map = list(range(10))
            
            # Create a transformed training set
            transformed_train = []
            for ex in task['train']:
                inp = np.array(ex['input'], dtype=np.uint8)
                transformed_inp = self._generate_spatial_transforms(inp)[transform_name]
                transformed_train.append({'input': transformed_inp, 'output': ex['output']})

            # Learn and validate mapping on the transformed set
            temp_color_map, confidence = self._learn_color_mapping(transformed_train)
            
            # Check if this transform consistently maps inputs to outputs
            score = 0
            for ex in transformed_train:
                predicted_output = self._apply_mapping_gpu(np.array(ex['input']), temp_color_map, np.array(ex['output']).shape)
                if np.array_equal(predicted_output, np.array(ex['output'])):
                    score += 1
            
            if score == len(task['train']): # Perfect match for all training pairs
                 if verbose:
                    print(f"[SPATIAL] Found consistent transform: {transform_name} with score {score}")
                 best_transform_name = transform_name
                 best_color_map = temp_color_map
                 break # Found a working transform

        if verbose:
            print(f"[SOLVER] Selected transform: '{best_transform_name}'")

        # --- APPLY TO TEST CASES ---
        predictions = []
        for test_case in task['test']:
            test_input = np.array(test_case['input'], dtype=np.uint8)
            
            # Apply the winning transformation
            transformed_test_input = self._generate_spatial_transforms(test_input)[best_transform_name]
            
            predicted_size = self._predict_output_size(task['train'], test_input.shape, best_transform_name)

            think_start = time.time()
            
            # Generate prediction using the best transform and its learned color map
            attempt1 = self._apply_mapping_gpu(transformed_test_input, best_color_map, predicted_size)
            
            think_time = time.time() - think_start
            if verbose:
                print(f"[THINK] GPU time: {think_time*1000:.2f}ms | Output: {attempt1.shape}")

            # For now, the second attempt is the same as the first
            predictions.append([
                attempt1.tolist(),
                attempt1.tolist() 
            ])

        total_time = time.time() - start
        if verbose:
            print(f"[TOTAL] Time: {total_time*1000:.1f}ms")

        return predictions

    def _render_quad(self, program):
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
        vbo = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.simple_vertex_array(program, vbo, 'in_vert')
        vao.render(moderngl.TRIANGLE_STRIP)
        vao.release()
        vbo.release()

    def get_stats(self):
        age = time.time() - self.birth_time
        return {
            'version': '9.1',
            'tasks_processed': self.tasks_processed,
            'successful_tasks': self.successful_tasks,
            'age_seconds': age,
            'memory_size': (self.memory_w, self.memory_h),
            'alive': True
        }

    def __del__(self):
        if hasattr(self, 'holographic_memory'): self.holographic_memory.release()
        if hasattr(self, 'ctx'): self.ctx.release()


# --- Global Brain Management ---
_global_brain_v9_1 = None

def get_brain_v9_1():
    global _global_brain_v9_1
    if _global_brain_v9_1 is None:
        _global_brain_v9_1 = LivingBrainV9_1()
    return _global_brain_v9_1

def solve_arc_task(task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
    brain = get_brain_v9_1()
    return brain.solve_task(task, verbose=verbose)


if __name__ == "__main__":
    print("="*80)
    print("CHIMERA v9.1 - LOCAL TEST")
    print("="*80)
    
    brain = get_brain_v9_1()

    # Test 1: Simple color mapping (from v9.0)
    task1 = {
        'train': [{'input': [[1, 2], [3, 4]], 'output': [[2, 3], [4, 5]]}],
        'test': [{'input': [[1, 2], [3, 4]]}]
    }
    print("\nTest 1: Simple color mapping (Identity transform)")
    result1 = brain.solve_task(task1)
    expected1 = [[2, 3], [4, 5]]
    print(f"Result:   {result1[0][0]}")
    print(f"Expected: {expected1}")
    print(f"Match: {result1[0][0] == expected1}")

    # Test 3: Rotation task
    task3 = {
        'train': [{'input': [[1, 0], [0, 0]], 'output': [[0, 1], [0, 0]]}],
        'test': [{'input': [[0, 2], [0, 0]]}]
    }
    print("\nTest 3: Spatial reasoning (rot270)")
    result3 = brain.solve_task(task3)
    expected3 = [[0, 0], [0, 2]] # rot270 of the test input
    print(f"Result:   {result3[0][0]}")
    print(f"Expected: {expected3}")
    print(f"Match: {result3[0][0] == expected3}")

    # Statistics
    stats = brain.get_stats()
    print("\n" + "="*80)
    print("v9.1 BRAIN STATISTICS")
    print("="*80)
    print(f"Version: {stats['version']}")
    print(f"Tasks processed: {stats['tasks_processed']}")
    print(f"Age: {stats['age_seconds']:.2f}s")
    print(f"Status: {'ALIVE' if stats['alive'] else 'DEAD'}")
    print("="*80)

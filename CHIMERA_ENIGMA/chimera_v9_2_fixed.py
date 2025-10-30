#!/usr/bin/env python3
"""
CHIMERA v9.2 - Object-Centric Reasoning Brain (Fixed)

Fixes a critical bug in the object reconstruction logic where the background
color was not preserved.

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import numpy as np
import moderngl
from typing import List, Dict, Tuple, Optional, Any
import time
from collections import Counter
from scipy.ndimage import label

# A simple class to hold object properties
class ArcObject:
    def __init__(self, grid: np.ndarray, color: int, position: Tuple[int, int]):
        self.grid = grid
        self.color = color
        self.position = position
        self.shape = grid.shape

    def __repr__(self):
        return f"Obj(color={self.color}, shape={self.shape}, pos={self.position})"

class LivingBrainV9_2_fixed:
    """
    Enhanced living brain with object detection and object-centric reasoning.
    """

    def __init__(self, memory_size=(256, 256)):
        """Birth of the enhanced brain"""
        print("="*80)
        print("CHIMERA v9.2 (Fixed) - OBJECT DETECTION BRAIN")
        print("="*80)
        print("Persistent GPU architecture with object-centric reasoning")
        print("="*80)

        self.ctx = moderngl.create_standalone_context()
        print(f"[GPU] {self.ctx.info['GL_RENDERER']}")

        self.memory_w, self.memory_h = memory_size
        self.holographic_memory = self.ctx.texture(size=(self.memory_w, self.memory_h), components=4, dtype='f4')
        zeros = np.zeros((self.memory_h, self.memory_w, 4), dtype=np.float32)
        self.holographic_memory.write(zeros.tobytes())
        print(f"[MEMORY] Holographic memory: {self.memory_w}x{self.memory_h}")

        self._compile_shaders()

        self.tasks_processed = 0
        self.birth_time = time.time()
        self.successful_tasks = 0
        print(f"[BRAIN] v9.2 (Fixed) awakened and ready")
        print("="*80)

    def _compile_shaders(self):
        vertex_shader = """#version 330\nin vec2 in_vert; out vec2 uv; void main() { gl_Position = vec4(in_vert, 0.0, 1.0); uv = (in_vert + 1.0) / 2.0; }"""
        fragment_shader_map = """#version 330
        uniform sampler2D u_input; uniform int u_color_map[10]; uniform ivec2 grid_size;
        in vec2 uv; out vec4 out_color;
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
        }"""
        self.map_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader_map)
        print("[SHADERS] Enhanced shaders compiled")

    def _generate_spatial_transforms(self, grid: np.ndarray) -> Dict[str, np.ndarray]:
        transforms = {
            'identity': grid, 'rot90': np.rot90(grid, 1), 'rot180': np.rot90(grid, 2), 'rot270': np.rot90(grid, 3),
        }
        flipped = np.fliplr(grid)
        transforms.update({
            'flip': flipped, 'flip_rot90': np.rot90(flipped, 1),
            'flip_rot180': np.rot90(flipped, 2), 'flip_rot270': np.rot90(flipped, 3)
        })
        return transforms

    def _detect_objects(self, grid: np.ndarray) -> List[ArcObject]:
        if grid.size == 0: return []
        colors, counts = np.unique(grid, return_counts=True)
        background_color = colors[np.argmax(counts)]
        objects = []
        for color in colors:
            if color == background_color: continue
            color_mask = (grid == color)
            labeled_array, num_features = label(color_mask)
            for i in range(1, num_features + 1):
                obj_mask = (labeled_array == i)
                coords = np.argwhere(obj_mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                obj_grid_slice = obj_mask[y_min:y_max+1, x_min:x_max+1]
                obj_grid = np.where(obj_grid_slice, color, background_color)
                objects.append(ArcObject(grid=obj_grid, color=color, position=(y_min, x_min)))
        return objects

    def solve_task(self, task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
        self.tasks_processed += 1
        if verbose: print(f"\n[v9.2-fixed] Task #{self.tasks_processed} | Age: {time.time() - self.birth_time:.1f}s")

        try:
            train_pairs = []
            for ex in task['train']:
                in_grid = np.array(ex['input'], dtype=np.uint8)
                out_grid = np.array(ex['output'], dtype=np.uint8)
                train_pairs.append({'input': self._detect_objects(in_grid), 'output': self._detect_objects(out_grid), 'out_grid': out_grid})
        except Exception as e:
            if verbose: print(f"[OBJECTS] Failed to analyze objects: {e}")
            return [[[[0]]]]

        best_rule = None
        for transform_name in self._generate_spatial_transforms(np.zeros((1,1))):
            is_consistent_rule = True
            for pair in train_pairs:
                if len(pair['input']) != len(pair['output']) or len(pair['input']) == 0:
                    is_consistent_rule = False; break
                
                out_colors, out_counts = np.unique(pair['out_grid'], return_counts=True)
                bg_color = out_colors[np.argmax(out_counts)]
                reconstructed_grid = np.full_like(pair['out_grid'], bg_color)

                for in_obj in pair['input']:
                    transformed_obj_grid = self._generate_spatial_transforms(in_obj.grid)[transform_name]
                    out_obj = next((o for o in pair['output'] if o.color == in_obj.color), None)

                    if out_obj is None or transformed_obj_grid.shape != out_obj.shape:
                        is_consistent_rule = False; break
                    
                    y, x = out_obj.position
                    h, w = transformed_obj_grid.shape
                    reconstructed_grid[y:y+h, x:x+w] = transformed_obj_grid

                if not is_consistent_rule or not np.array_equal(reconstructed_grid, pair['out_grid']):
                    is_consistent_rule = False; break
            
            if is_consistent_rule:
                best_rule = {'type': 'spatial_transform_on_objects', 'transform_name': transform_name}
                if verbose: print(f"[SOLVER] Found consistent rule: Transform all objects with '{transform_name}'")
                break

        if not best_rule:
            if verbose: print("[SOLVER] No consistent object-based rule found.")
            return [[[[0]]]]

        predictions = []
        for test_case in task['test']:
            test_input_grid = np.array(test_case['input'], dtype=np.uint8)
            test_in_objects = self._detect_objects(test_input_grid)
            
            in_colors, in_counts = np.unique(test_input_grid, return_counts=True)
            bg_color = in_colors[np.argmax(in_counts)]
            final_grid = np.full_like(test_input_grid, bg_color)
            
            if best_rule['type'] == 'spatial_transform_on_objects':
                transform = best_rule['transform_name']
                for obj in test_in_objects:
                    transformed_grid = self._generate_spatial_transforms(obj.grid)[transform]
                    y, x = obj.position # Simplistic placement, assumes objects don't move
                    h, w = transformed_grid.shape
                    if y+h <= final_grid.shape[0] and x+w <= final_grid.shape[1]:
                         final_grid[y:y+h, x:x+w] = transformed_grid

            predictions.append([final_grid.tolist()])
        return predictions

    def get_stats(self):
        return {'version': '9.2-fixed', 'tasks_processed': self.tasks_processed, 'age_seconds': time.time() - self.birth_time}

# --- Global Brain Management ---
_global_brain_v9_2_fixed = None
def get_brain_v9_2_fixed():
    global _global_brain_v9_2_fixed
    if _global_brain_v9_2_fixed is None: _global_brain_v9_2_fixed = LivingBrainV9_2_fixed()
    return _global_brain_v9_2_fixed

def solve_arc_task(task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
    return get_brain_v9_2_fixed().solve_task(task, verbose=verbose)


if __name__ == "__main__":
    print("="*80)
    print("CHIMERA v9.2 (Fixed) - LOCAL TEST")
    print("="*80)
    brain = get_brain_v9_2_fixed()

    task1 = {
        'train': [{
            'input': [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
            'output': [[0, 1, 1], [0, 1, 0], [0, 0, 0]]
        }],
        'test': [{
            'input': [[0, 0, 0], [2, 2, 0], [0, 2, 0]]
        }]
    }
    print("\nTest 1: Object Rotation (rot90)")
    result1 = brain.solve_task(task1)
    expected1 = [[0, 0, 0], [0, 2, 2], [0, 2, 0]]
    print(f"Result:   {result1[0][0]}")
    print(f"Expected: {expected1}")
    print(f"Match: {result1[0][0] == expected1}")

    stats = brain.get_stats()
    print("\n" + "="*80)
    print(f"v{stats['version']} BRAIN STATISTICS")
    print("="*80)
    print(f"Tasks processed: {stats['tasks_processed']}")
    print(f"Age: {stats['age_seconds']:.2f}s")
    print("="*80)

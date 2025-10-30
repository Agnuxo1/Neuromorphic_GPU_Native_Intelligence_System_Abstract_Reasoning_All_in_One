#!/usr/bin/env python3
"""
CHIMERA v9.3 - Spatio-Temporal Reasoning Brain

Extends v9.2 by adding object displacement (translation) detection.
The brain can now understand rules like "rotate object X and move it Y steps down".

Francisco Angulo de Lafuente - CHIMERA Project 2025
"""

import numpy as np
import moderngl
from typing import List, Dict, Tuple, Optional, Any
import time
from collections import Counter
from scipy.ndimage import label

class ArcObject:
    def __init__(self, grid: np.ndarray, color: int, position: Tuple[int, int]):
        self.grid = grid
        self.color = color
        self.position = position
        self.shape = grid.shape

    def __repr__(self):
        return f"Obj(color={self.color}, shape={self.shape}, pos={self.position})"

class LivingBrainV9_3:
    """
    Brain with object detection, spatial transformation, and displacement reasoning.
    """

    def __init__(self, memory_size=(256, 256)):
        print("="*80)
        print("CHIMERA v9.3 - OBJECT TRANSLATION BRAIN")
        print("="*80)
        self.ctx = moderngl.create_standalone_context()
        print(f"[GPU] {self.ctx.info['GL_RENDERER']}")
        # ... (rest of init is standard)
        self.memory_w, self.memory_h = memory_size
        self.holographic_memory = self.ctx.texture(size=(self.memory_w, self.memory_h), components=4, dtype='f4')
        self.tasks_processed = 0
        self.birth_time = time.time()
        print(f"[BRAIN] v9.3 awakened and ready")
        print("="*80)

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
                obj_mask_full = (labeled_array == i)
                coords = np.argwhere(obj_mask_full)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                obj_mask_slice = obj_mask_full[y_min:y_max+1, x_min:x_max+1]
                obj_grid = np.where(obj_mask_slice, color, -1) # Use -1 for transparency
                objects.append(ArcObject(grid=obj_grid, color=color, position=(y_min, x_min)))
        return objects

    def solve_task(self, task: Dict, verbose: bool = True) -> List[List[List[List[int]]]]:
        self.tasks_processed += 1
        if verbose: print(f"[v9.3] Task #{self.tasks_processed} | Age: {time.time() - self.birth_time:.1f}s")

        try:
            train_pairs = [{'input_objs': self._detect_objects(np.array(ex['input'])), 'output_objs': self._detect_objects(np.array(ex['output']))} for ex in task['train']]
        except Exception as e:
            if verbose: print(f"[OBJECTS] Failed to analyze objects: {e}")
            return [[[[0]]]]

        best_rule = None
        for transform_name in self._generate_spatial_transforms(np.zeros((1,1))):
            rule_displacement = None
            is_rule_consistent = True
            for pair in train_pairs:
                if len(pair['input_objs']) != len(pair['output_objs']) or not pair['input_objs']:
                    is_rule_consistent = False; break
                
                pair_displacement = None
                for in_obj in pair['input_objs']:
                    transformed_grid = self._generate_spatial_transforms(in_obj.grid)[transform_name]
                    out_obj = next((o for o in pair['output_objs'] if o.color == in_obj.color and transformed_grid.shape == o.shape and np.array_equal(self._generate_spatial_transforms(in_obj.grid)[transform_name], o.grid) ), None)
                    
                    if out_obj is None:
                        is_rule_consistent = False; break

                    current_displacement = (out_obj.position[0] - in_obj.position[0], out_obj.position[1] - in_obj.position[1])
                    if pair_displacement is None: pair_displacement = current_displacement
                    if pair_displacement != current_displacement:
                        is_rule_consistent = False; break
                
                if not is_rule_consistent: break
                if rule_displacement is None: rule_displacement = pair_displacement
                if rule_displacement != pair_displacement:
                    is_rule_consistent = False; break
            
            if is_rule_consistent:
                best_rule = {'transform': transform_name, 'displacement': rule_displacement}
                if verbose: print(f"[SOLVER] Found consistent rule: transform '{transform_name}', move by {rule_displacement}")
                break

        if not best_rule:
            if verbose: print("[SOLVER] No consistent object-based rule found.")
            return [[[[0]]]]

        predictions = []
        for test_case in task['test']:
            test_input_grid = np.array(test_case['input'], dtype=np.uint8)
            test_in_objects = self._detect_objects(test_input_grid)
            
            # Assume output size is same as input for now
            final_grid = np.copy(test_input_grid)
            
            transform = best_rule['transform']
            disp = best_rule['displacement']
            
            # Create a blank grid with original background to draw on
            colors, counts = np.unique(test_input_grid, return_counts=True)
            bg_color = colors[np.argmax(counts)]
            final_grid = np.full_like(test_input_grid, bg_color)

            for obj in test_in_objects:
                transformed_grid = self._generate_spatial_transforms(obj.grid)[transform]
                new_y, new_x = obj.position[0] + disp[0], obj.position[1] + disp[1]
                h, w = transformed_grid.shape
                if 0 <= new_y and new_y + h <= final_grid.shape[0] and 0 <= new_x and new_x + w <= final_grid.shape[1]:
                    patch = final_grid[new_y:new_y+h, new_x:new_x+w]
                    patch[transformed_grid != -1] = transformed_grid[transformed_grid != -1]

            predictions.append([final_grid.tolist()])
        return predictions

if __name__ == "__main__":
    print("="*80)
    print("CHIMERA v9.3 - LOCAL TEST")
    print("="*80)
    brain = LivingBrainV9_3()

    task1 = {
        'train': [{
            'input': [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
            'output': [[0, 0, 0], [0, 1, 1], [0, 0, 1]]
        }],
        'test': [{
            'input': [[0, 2, 0], [2, 2, 0], [0, 0, 0]]
        }]
    }
    print("\nTest 1: Object Rotation (rot270) and Translation (1, 1)")
    result1 = brain.solve_task(task1)
    expected1 = [[0, 0, 0], [0, 0, 2], [0, 2, 2]]
    print(f"Result:   {result1[0][0]}")
    print(f"Expected: {expected1}")
    print(f"Match: {result1[0][0] == expected1}")
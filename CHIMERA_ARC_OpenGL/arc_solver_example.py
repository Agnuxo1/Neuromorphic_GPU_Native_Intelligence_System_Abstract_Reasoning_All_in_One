#!/usr/bin/env python3
"""
CHIMERA v10.0 - ARC-AGI Task Loader and Solver

Load tasks from JSON files (ARC-AGI format) and generate solutions.
Compatible with Kaggle submission format.

Usage:
    python arc_solver_example.py --input training/ --output predictions.json
    python arc_solver_example.py --task-file task_001.json
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List
import time

from chimera_v10_0 import LivingBrainV10, solve_arc_task


class ARCTaskLoader:
    """
    Load and manage ARC-AGI tasks from JSON files.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: Directory containing JSON task files
        """
        self.data_dir = data_dir
        self.tasks = {}
    
    def load_task(self, filepath: str) -> Dict:
        """Load a single task from JSON file."""
        with open(filepath, 'r') as f:
            task = json.load(f)
        
        # ARC tasks are stored as {task_id: {train: [...], test: [...]}}
        # or directly as {train: [...], test: [...]}
        
        if 'train' in task and 'test' in task:
            return task
        else:
            # Assume it's a dict of tasks
            task_id = list(task.keys())[0]
            return task[task_id]
    
    def load_directory(self, directory: str) -> Dict[str, Dict]:
        """Load all JSON tasks from directory."""
        directory = Path(directory)
        
        tasks = {}
        for json_file in directory.glob('*.json'):
            task_id = json_file.stem
            
            try:
                with open(json_file, 'r') as f:
                    content = json.load(f)
                
                # Handle different JSON formats
                if 'train' in content and 'test' in content:
                    tasks[task_id] = content
                elif isinstance(content, dict):
                    # File contains multiple tasks
                    for tid, task_data in content.items():
                        tasks[tid] = task_data
            
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        
        self.tasks = tasks
        return tasks
    
    def get_task(self, task_id: str) -> Dict:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self) -> List[str]:
        """List all loaded task IDs."""
        return list(self.tasks.keys())


class KaggleSubmissionFormatter:
    """
    Format predictions for Kaggle submission.
    """
    
    @staticmethod
    def format_submission(predictions: Dict[str, List]) -> List[Dict]:
        """
        Format predictions into Kaggle submission format.
        
        Args:
            predictions: {task_id: [attempt1, attempt2]}
        
        Returns:
            List of dicts in Kaggle format
        """
        submission = []
        
        for task_id, attempts in predictions.items():
            task_predictions = {}
            
            # Each test case gets 2 attempts
            for test_idx, (attempt1, attempt2) in enumerate(attempts):
                task_predictions[test_idx] = [attempt1, attempt2]
            
            submission.append({task_id: task_predictions})
        
        return submission
    
    @staticmethod
    def save_submission(submission: List[Dict], filepath: str):
        """Save submission to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(submission, f, indent=2)
        
        print(f"Submission saved to: {filepath}")


class ARCSolver:
    """
    Main solver class for batch processing.
    """
    
    def __init__(self, verbose: bool = True, max_tasks: int = None):
        """
        Args:
            verbose: Print progress
            max_tasks: Maximum number of tasks to solve (None = all)
        """
        self.brain = LivingBrainV10()
        self.verbose = verbose
        self.max_tasks = max_tasks
        
        self.stats = {
            'tasks_attempted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_time': 0.0
        }
    
    def solve_task(self, task: Dict, task_id: str = None) -> List:
        """
        Solve single task.
        
        Returns:
            List of [attempt1, attempt2] for each test case
        """
        if self.verbose and task_id:
            print(f"\n{'='*60}")
            print(f"Solving task: {task_id}")
            print(f"{'='*60}")
        
        try:
            start = time.time()
            predictions = self.brain.solve_task(task, verbose=self.verbose)
            elapsed = time.time() - start
            
            self.stats['tasks_completed'] += 1
            self.stats['total_time'] += elapsed
            
            if self.verbose:
                print(f"✓ Completed in {elapsed:.2f}s")
            
            return predictions
        
        except Exception as e:
            self.stats['tasks_failed'] += 1
            print(f"✗ Failed: {str(e)}")
            
            # Return empty predictions
            num_test_cases = len(task.get('test', []))
            return [[[0]], [[0]]] * num_test_cases
    
    def solve_batch(self, tasks: Dict[str, Dict]) -> Dict[str, List]:
        """
        Solve multiple tasks.
        
        Args:
            tasks: {task_id: task_data}
        
        Returns:
            {task_id: predictions}
        """
        results = {}
        
        task_ids = list(tasks.keys())
        if self.max_tasks:
            task_ids = task_ids[:self.max_tasks]
        
        total = len(task_ids)
        
        print(f"\n{'='*60}")
        print(f"Solving {total} tasks...")
        print(f"{'='*60}\n")
        
        for idx, task_id in enumerate(task_ids, 1):
            self.stats['tasks_attempted'] += 1
            
            print(f"\n[{idx}/{total}] Task: {task_id}")
            
            task = tasks[task_id]
            predictions = self.solve_task(task, task_id)
            results[task_id] = predictions
        
        return results
    
    def print_stats(self):
        """Print solving statistics."""
        print(f"\n{'='*60}")
        print("STATISTICS")
        print(f"{'='*60}")
        print(f"Tasks attempted:  {self.stats['tasks_attempted']}")
        print(f"Tasks completed:  {self.stats['tasks_completed']}")
        print(f"Tasks failed:     {self.stats['tasks_failed']}")
        print(f"Total time:       {self.stats['total_time']:.1f}s")
        
        if self.stats['tasks_completed'] > 0:
            avg_time = self.stats['total_time'] / self.stats['tasks_completed']
            print(f"Average per task: {avg_time:.2f}s")
        
        print(f"{'='*60}\n")


def visualize_grid(grid: List[List[int]], title: str = "Grid"):
    """
    Simple ASCII visualization of grid.
    """
    print(f"\n{title}:")
    for row in grid:
        print(" ".join(str(c) for c in row))


def compare_predictions(task: Dict, predictions: List):
    """
    Compare predictions with expected output (if available).
    """
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)
    
    for test_idx, test_case in enumerate(task['test']):
        print(f"\nTest case {test_idx + 1}:")
        
        # Show input
        visualize_grid(test_case['input'], "Input")
        
        # Show predictions
        if test_idx < len(predictions):
            attempt1, attempt2 = predictions[test_idx]
            visualize_grid(attempt1, "Attempt 1")
            visualize_grid(attempt2, "Attempt 2")
        
        # Show expected output if available
        if 'output' in test_case:
            visualize_grid(test_case['output'], "Expected")
            
            # Check if any attempt matches
            if test_idx < len(predictions):
                match1 = np.array_equal(attempt1, test_case['output'])
                match2 = np.array_equal(attempt2, test_case['output'])
                
                if match1 or match2:
                    print("✓ At least one attempt matches expected output!")
                else:
                    print("✗ Neither attempt matches expected output")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CHIMERA v10.0 - ARC-AGI Task Solver"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input directory with JSON task files or single task file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output JSON file for predictions'
    )
    
    parser.add_argument(
        '--task-id',
        type=str,
        help='Solve specific task by ID'
    )
    
    parser.add_argument(
        '--max-tasks',
        type=int,
        help='Maximum number of tasks to solve'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization of predictions'
    )
    
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Disable verbose output'
    )
    
    args = parser.parse_args()
    
    # If no args, run example
    if not args.input:
        print("No input provided. Running example task...")
        run_example()
        return
    
    # Initialize
    loader = ARCTaskLoader()
    solver = ARCSolver(
        verbose=not args.no_verbose,
        max_tasks=args.max_tasks
    )
    formatter = KaggleSubmissionFormatter()
    
    # Load tasks
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single task file
        task = loader.load_task(str(input_path))
        task_id = args.task_id or input_path.stem
        tasks = {task_id: task}
    elif input_path.is_dir():
        # Directory of tasks
        tasks = loader.load_directory(str(input_path))
    else:
        print(f"Error: {args.input} not found")
        return
    
    print(f"Loaded {len(tasks)} tasks")
    
    # Solve specific task if requested
    if args.task_id and args.task_id in tasks:
        task = tasks[args.task_id]
        predictions = solver.solve_task(task, args.task_id)
        
        if args.visualize:
            compare_predictions(task, predictions)
        
        results = {args.task_id: predictions}
    else:
        # Solve all tasks
        results = solver.solve_batch(tasks)
    
    # Format and save submission
    submission = formatter.format_submission(results)
    formatter.save_submission(submission, args.output)
    
    # Print stats
    solver.print_stats()


def run_example():
    """Run example task."""
    print("\n" + "="*60)
    print("CHIMERA v10.0 - EXAMPLE TASK")
    print("="*60)
    
    # Example task: Color mapping
    task = {
        'train': [
            {
                'input': [[0, 1, 2], [1, 1, 2], [2, 2, 0]],
                'output': [[0, 3, 4], [3, 3, 4], [4, 4, 0]]
            },
            {
                'input': [[1, 0, 1], [2, 2, 2], [0, 1, 2]],
                'output': [[3, 0, 3], [4, 4, 4], [0, 3, 4]]
            }
        ],
        'test': [
            {
                'input': [[1, 1, 0], [2, 1, 2], [0, 2, 2]]
            }
        ]
    }
    
    print("\nTraining Examples:")
    for i, example in enumerate(task['train'], 1):
        print(f"\nExample {i}:")
        visualize_grid(example['input'], "  Input")
        visualize_grid(example['output'], "  Output")
    
    print("\nTest Input:")
    visualize_grid(task['test'][0]['input'])
    
    # Solve
    print("\nSolving...")
    solver = ARCSolver(verbose=True)
    predictions = solver.solve_task(task, "example")
    
    # Show predictions
    print("\nPredictions:")
    visualize_grid(predictions[0][0], "Attempt 1")
    visualize_grid(predictions[0][1], "Attempt 2")
    
    # Expected: 1→3, 2→4, 0→0
    print("\nExpected mapping: 1→3, 2→4, 0→0")
    expected = [[3, 3, 0], [4, 3, 4], [0, 4, 4]]
    visualize_grid(expected, "Expected Output")
    
    # Check
    match1 = np.array_equal(predictions[0][0], expected)
    match2 = np.array_equal(predictions[0][1], expected)
    
    if match1 or match2:
        print("\n✓ SUCCESS! At least one attempt matches expected output")
    else:
        print("\n✗ Neither attempt matches (but that's ok for complex tasks)")
    
    solver.print_stats()


if __name__ == "__main__":
    main()

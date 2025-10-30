import json
import numpy as np
import time
from chimera_v10_0 import solve_arc_task, LivingBrainV10, hungarian_color_mapping

def compute_confidence(color_map, train_examples):
    """Compute confidence in color mapping."""
    if color_map == list(range(10)):
        return 1.0

    total = 0
    consistent = 0

    for ex in train_examples:
        inp = np.array(ex['input'], dtype=np.uint8)
        out = np.array(ex['output'], dtype=np.uint8)

        if inp.shape != out.shape:
            continue

        for y in range(inp.shape[0]):
            for x in range(inp.shape[1]):
                old_c = int(inp[y, x])
                expected_c = color_map[old_c]
                actual_c = int(out[y, x])

                total += 1
                if expected_c == actual_c:
                    consistent += 1

    return consistent / max(1, total)

def calculate_accuracy(tasks, solutions):
    """Calculates the accuracy of the model."""
    total_tasks = 0
    tasks_solved = 0
    
    brain = LivingBrainV10()

    for task_name, task in tasks.items():
        if task_name not in solutions:
            continue
            
        total_tasks += 1
        
        color_map = hungarian_color_mapping(task['train'])
        confidence = compute_confidence(color_map, task['train'])
        
        print(f"Task: {task_name}, Confidence: {confidence:.2f}")
        
        task_solutions = solutions[task_name]
        predictions = brain.solve_task(task, verbose=False)
        
        all_test_cases_solved = True
        if not task['test']:
            all_test_cases_solved = False

        for i, test_case in enumerate(task['test']):
            prediction_pair = predictions[i]
            attempt1 = np.array(prediction_pair[0])
            attempt2 = np.array(prediction_pair[1])
            solution = np.array(task_solutions[i])

            if not (np.array_equal(attempt1, solution) or np.array_equal(attempt2, solution)):
                all_test_cases_solved = False
                break
        
        if all_test_cases_solved:
            tasks_solved += 1

    if total_tasks > 0:
        overall_accuracy = tasks_solved / total_tasks
    else:
        overall_accuracy = 0.0
        
    return overall_accuracy

if __name__ == "__main__":
    with open("d:\\ARC2_CHIMERA\\CHIMERA_ARC_OpenGL\\Data\\arc-agi_evaluation_challenges.json", 'r') as f:
        tasks = json.load(f)
        
    with open("d:\\ARC2_CHIMERA\\CHIMERA_ARC_OpenGL\\Data\\arc-agi_evaluation_solutions.json", 'r') as f:
        solutions = json.load(f)
    
    print("Calculating accuracy...")
    accuracy = calculate_accuracy(tasks, solutions)
    print(f"Accuracy: {accuracy:.2%}")

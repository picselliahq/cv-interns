import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_and_save_cases(evaluations, top_n=3, worst_n=3, save_dir='evaluation_cases'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Sort evaluations by some performance metric (e.g., accuracy, IoU, etc.)
    # Assuming 'evaluations' is a list of dictionaries with a 'score' key
    evaluations_sorted = sorted(evaluations, key=lambda x: x['score'], reverse=True)

    # Get the top N and worst N cases
    top_cases = evaluations_sorted[:top_n]
    worst_cases = evaluations_sorted[-worst_n:]

    # Function to draw rectangles on the image
    def draw_rectangles(ax, rectangles, color):
        for (x, y, w, h) in rectangles:
            rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    # Function to process and save a single case
    def process_case(case, case_type, index):
        fig, ax = plt.subplots(1)
        # Load image
        image_path = case['image_path']
        image = plt.imread(image_path)
        ax.imshow(image)

        # Draw predictions and ground truth
        draw_rectangles(ax, case['predictions'], 'r')
        draw_rectangles(ax, case['ground_truth'], 'g')

        # Save the image
        case_save_path = os.path.join(save_dir, f'{case_type}_case_{index}.png')
        plt.savefig(case_save_path)
        plt.close()

    # Process and save top cases
    for i, case in enumerate(top_cases):
        process_case(case, 'top', i)

    # Process and save worst cases
    for i, case in enumerate(worst_cases):
        process_case(case, 'worst', i)

# Assuming 'experiment' is an instance of a Picsellia Experiment object
# and 'client' is an instance of a Picsellia Client object
def list_and_draw_cases(experiment, client):
    evaluations = experiment.list_evaluations()
    # Here we would need to extract the necessary information from each evaluation
    # For the sake of this example, let's assume we have a function that does this
    # and returns a list of dictionaries with keys 'score', 'image_path', 'predictions', and 'ground_truth'
    evaluation_data = [extract_evaluation_data(eval_obj, client) for eval_obj in evaluations]
    draw_and_save_cases(evaluation_data)

# Placeholder for the actual data extraction function
def extract_evaluation_data(eval_obj, client):
    # This function would interact with the Picsellia SDK to extract the necessary data
    # For now, we'll return a mock dictionary
    return {
        'score': eval_obj.score,  # Placeholder for actual score extraction
        'image_path': '/path/to/image.png',  # Placeholder for actual image path extraction
        'predictions': [(10, 10, 100, 100)],  # Placeholder for actual predictions extraction
        'ground_truth': [(15, 15, 90, 90)]  # Placeholder for actual ground truth extraction
    }
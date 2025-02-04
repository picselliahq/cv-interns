from smolagents import Tool, ToolCallingAgent, HfApiModel, CodeAgent, ManagedAgent, GradioUI, LiteLLMModel
from picsellia import Datalake, Tag, DatasetVersion, Client
from picsellia.sdk.data import MultiData
from typing import List, Union
import os
from picsellia.exceptions import ResourceNotFoundError, NoBaseModelVersionError
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def calculate_average_metrics(evaluations: List[dict]):
    """
    Calculate average precision and recall metrics for each label across a list of evaluation objects.

    This function takes a list of evaluation objects, which contain information about labeled rectangles and false negatives for each class. It calculates the precision, recall, true positives, false positives, and false negatives for each label and returns a dictionary with these metrics.

    Args:
        evaluations (list): 
            A list of evaluation objects, where each object is a dictionary with the following keys:
                - 'rectangles' (list): A list of rectangles. Each rectangle is a dictionary containing:
                    - 'label' (dict): The label information with a key 'name' for the label name.
                    - 'false_positive' (bool): A boolean indicating if the rectangle is a false positive.
                - 'false_negatives_by_class' (dict): A dictionary mapping label names to the number of false negatives for that label.

    Returns:
        dict: A dictionary where:
            - Keys are label names (str).
            - Values are dictionaries containing:
                - 'average_precision' (float): The precision value for the label.
                - 'average_recall' (float): The recall value for the label.
                - 'true_positives' (int): The number of true positives for the label.
                - 'false_positives' (int): The number of false positives for the label.
                - 'false_negatives' (int): The number of false negatives for the label.

    Example:
        Input:
            evaluations = [
                {
                    'rectangles': [
                        {'label': {'name': 'car'}, 'false_positive': False},
                        {'label': {'name': 'car'}, 'false_positive': True},
                        {'label': {'name': 'tree'}, 'false_positive': False}
                    ],
                    'false_negatives_by_class': {'car': 3, 'tree': 3}
                }
            ]

        Output:
            {
                'car': {
                    'average_precision': 0.6666666666666666,
                    'average_recall': 0.4,
                    'true_positives': 2,
                    'false_positives': 1,
                    'false_negatives': 3
                },
                'tree': {
                    'average_precision': 0.5,
                    'average_recall': 0.25,
                    'true_positives': 1,
                    'false_positives': 1,
                    'false_negatives': 3
                }
            }
    """
    from collections import defaultdict

    # Initialize data storage
    metrics_per_label = defaultdict(lambda: {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'precision_sum': 0,
        'recall_sum': 0,
        'count': 0
    })

    # Iterate over all evaluations
    for eval_obj in evaluations:
        for rectangle in eval_obj.get('rectangles', []):
            label_name = rectangle['label']['name']
            is_false_positive = rectangle.get('false_positive', False)
            is_true_positive = not is_false_positive

            # Update metrics for this label
            if is_true_positive:
                metrics_per_label[label_name]['true_positives'] += 1
            else:
                metrics_per_label[label_name]['false_positives'] += 1

        # Calculate false negatives per evaluation
        # Assuming false negatives are stored in the evaluation object as 'false_negatives_by_class'
        false_negatives_by_class = eval_obj.get('false_negatives_by_class', {})
        for label_name, false_negatives in false_negatives_by_class.items():
            metrics_per_label[label_name]['false_negatives'] += false_negatives

    # Calculate averages
    average_metrics = {}
    for label, metrics in metrics_per_label.items():
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        average_metrics[label] = {
            'average_precision': precision,
            'average_recall': recall,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    return average_metrics



class AnalyzePicselliaProject(Tool):
    
    name = "analyze_picsellia_project"
    description = """
    Analyzes the specified Picsellia project and compiles information about its experiments.
    Returns:
        dict: A dictionary containing an overview of the project and detailed information
                about each experiment, including dataset overviews, attachments, logs, and
                calculated performance metrics.
    """
    inputs = {
        "project_name": {
            "type": "string", 
            "description": "Name of the Picsellia project to analyze",
        }
    }
    output_type = "object"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Client(api_token=os.environ["PICSELLIA_TOKEN"])

    def forward(self, project_name: str) -> dict:
        """
        Analyzes the specified Picsellia project and compiles information about its experiments.

        Args:
            project_name (str): The name of the Picsellia project to analyze.

        Returns:
            dict: A dictionary containing an overview of the project and detailed information
                  about each experiment, including dataset overviews, attachments, logs, and
                  calculated performance metrics.
        """
        project = self.client.get_project(project_name=project_name)
        experiments = project.list_experiments()

        dump = {
            "project_overview": project.sync(),
            "experiments": []
        }

        for experiment in experiments:
            dataset_overview = {}
            for dsv in experiment.list_attached_dataset_versions():
                dataset_overview[dsv.version + " dataset"] = {
                    "name": dsv.name,
                    "labels": [e.name for e in dsv.list_labels()],
                    "metadata": dsv.retrieve_stats().model_dump(),
                    "images_count": dsv.sync()["size"]
                }

            for experiment in experiments:
                # Get all attachments
                try:
                    base_model = experiment.get_base_model_version()
                except NoBaseModelVersionError as e:
                    base_model = experiment.get_base_experiment().get_base_model_version()

                attachments = {
                    'base_architecture_trained': f"{base_model.name}/{base_model.version}",
                    'task': experiment.list_attached_dataset_versions()[0].type.value,
                    'logs': [],
                    'other_files': []
                }
                # Get any stored files/artifacts
                try:
                    stored_files = experiment.list_stored_files()
                    if stored_files:
                        attachments['other_files'].extend(stored_files)
                except:
                    pass
                # Get all logs
                logs = {
                    'metrics': {},
                    'parameters': {},
                    'other_logs': {}
                }
                # Retrieve all experiment logs
                all_logs = experiment.list_logs()
                for log in all_logs:
                    log_name = log.name.lower()

                    # Categorize logs based on common names
                    if log_name in ['accuracy', 'loss', 'train-split', 'test-split', 'eval-split']:
                        logs['metrics'][log_name] = log.data
                    elif log_name == 'parameters':
                        logs['parameters'] = log.data
                    elif log_name == 'labelmap':
                        logs['labelmap'] = log.data
                    else:
                        logs['other_logs'][log_name] = log.data

                evaluations = experiment.list_evaluations()
                # Sync each evaluation to get latest data
                synced_evaluations = []
                for eval_obj in evaluations:
                    synced_eval = eval_obj.sync()
                    synced_evaluations.append(synced_eval)

                # Calculate metrics
                metrics = calculate_average_metrics(synced_evaluations)
                dump["experiments"].append({
                    "experiment": experiment.sync(),
                    "dataset_overview": dataset_overview,
                    "metadata": {
                        "metadata": logs,
                        "performances": metrics
                    }
                })
        return dump




# Example usage:
# client = Client(api_token="your_api_token")
# experiment = client.get_experiment("your_experiment_id")
# list_and_draw_cases(experiment, client)

            
model_name = "meta-llama/Llama-3.3-70B-Instruct"
model = HfApiModel(model_name, token=os.getenv("HF_TOKEN"))
project_analyzer_agent = CodeAgent(
    model=model,
    tools=[AnalyzePicselliaProject()],

)


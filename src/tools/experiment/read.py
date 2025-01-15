from smolagents import Tool
from picsellia import Client, Experiment
from picsellia.sdk.evaluation import MultiEvaluation
from typing import List

class ListEvaluationsTool(Tool):
    """Tool for listing all evaluations in a Picsellia Experiment.

    This tool retrieves all evaluations that have been performed on a given experiment,
    including metrics and results.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The authenticated Picsellia client instance.
        experiment_id (str):
            - Type: `string`
            - Description: ID of the Picsellia experiment to get evaluations from.

    Output:
        output_type (list):
            A list of evaluation objects from the experiment containing evaluation metrics and results.

    Example:
        ```python
        client = some_client_instance
        experiment_id = "experiment-123"
        evaluations = tool.forward(client, experiment_id)
        for eval in evaluations:
            print(f"Evaluation: {eval.name}, Metrics: {eval.metrics}")
        ```
    """
    name = "list_evaluations"
    description = """
    This tool lists all evaluations performed on a specified Picsellia experiment.
    It returns a list of evaluation objects containing metrics and results.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "experiment_id": {
            "type": "string",
            "description": "ID of the Picsellia experiment to get evaluations from",
        }
    }
    output_type = "object"

    def forward(self, client: Client, experiment_id: str) -> List:
        """
        Lists all evaluations for the specified experiment.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            experiment_id (str): ID of the experiment to get evaluations from
            
        Returns:
            List: List of evaluation objects
            
        Raises:
            ValueError: If evaluations cannot be retrieved or other errors occur
        """
        try:
            experiment = client.get_experiment_by_id(experiment_id)
            return [e.sync() for e in experiment.list_evaluations()]
        except Exception as e:
            raise ValueError(f"Failed to list evaluations: {str(e)}")

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


class ListEvaluationsAndMetricsTool(Tool):
    """Tool for listing evaluations and calculating metrics in a Picsellia Experiment.

    This tool retrieves all evaluations from a Picsellia experiment, syncs them to get the latest data,
    and calculates precision/recall metrics for each label.

    Inputs:
        client (Client):
            - Type: `object` 
            - Description: The authenticated Picsellia client instance.
        experiment_id (str):
            - Type: `string`
            - Description: ID of the Picsellia experiment to get evaluations from.

    Output:
        output_type (dict):
            A dictionary containing metrics per label including:
            - average_precision: The precision value for the label
            - average_recall: The recall value for the label
            - true_positives: Number of true positives
            - false_positives: Number of false positives  
            - false_negatives: Number of false negatives
    """
    name = "list_evaluations_and_metrics_by_label"
    description = """
    This tool lists all evaluations from a Picsellia experiment and calculates average metrics
    like precision and recall for each label.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "experiment_id": {
            "type": "string", 
            "description": "ID of the Picsellia experiment to get evaluations from",
        }
    }
    output_type = "object"

    def forward(self, client: Client, experiment_id: str) -> dict:
        """
        Lists evaluations and calculates metrics for the specified experiment.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            experiment_id (str): ID of the experiment to get evaluations from
            
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
        Raises:
            ValueError: If evaluations cannot be retrieved or metrics calculation fails
        """
        try:
            # Get experiment and list evaluations
            experiment = client.get_experiment_by_id(experiment_id)
            evaluations = experiment.list_evaluations()
            
            # Sync each evaluation to get latest data
            synced_evaluations = []
            for eval_obj in evaluations:
                synced_eval = eval_obj.sync()
                synced_evaluations.append(synced_eval)
                
            # Calculate metrics
            metrics = calculate_average_metrics(synced_evaluations)
            
            return metrics
            
        except Exception as e:
            raise ValueError(f"Failed to get evaluations and calculate metrics: {str(e)}")




list_evaluations_and_metrics_by_label = ListEvaluationsAndMetricsTool()





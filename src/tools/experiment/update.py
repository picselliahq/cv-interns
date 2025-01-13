import picsellia 
from smolagents import Tool
from picsellia import Client, Experiment


class UpdateExperimentParametersTool(Tool):
    """
    A tool to update the hyperparameters or other configurable parameters of an existing experiment in Picsellia.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The Picsellia SDK client instance used to interact with the API.
        experiment_id (str):
            - Type: `string`
            - Description: The unique ID of the experiment to update.
        parameters (dict):
            - Type: `dict`
            - Description: A dictionary of parameters to update for the experiment. The keys are parameter names, and the values are their new values.

    Output:
        output_type (Experiment): 
            The updated `Experiment` object.

    Example:
        ```
        # Example usage
        client = Client(api_token="your_api_token", organization_name="your_organization_name")
        experiment_id = "12345"
        updated_parameters = {
            "learning_rate": 0.001,
            "batch_size": 32,
        }

        # Update the experiment's parameters
        updated_experiment = tool.forward(
            client=client,
            experiment_id=experiment_id,
            parameters=updated_parameters
        )
        print(updated_experiment.parameters)
        ```

    Notes:
        - The tool uses the `get_experiment` and `update_parameters` methods from the Picsellia SDK.
        - Only the parameters provided in the input dictionary are updated; other parameters remain unchanged.
        - Proper error handling ensures that issues during update operations are communicated effectively.
    """
    name = "update_experiment_parameters"
    description = """
    This tool updates the hyperparameters or other configurable parameters of an existing experiment in Picsellia. 
    It allows fine-tuning of experiment parameters after creation.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia SDK client instance used to interact with the API.",
        },
        "experiment_id": {
            "type": "string",
            "description": "The unique ID of the experiment to update.",
        },
        "parameters": {
            "type": "object",
            "description": "A dictionary of parameters to update for the experiment. The keys are parameter names, and the values are their new values.",
        },
    }
    output_type = "object"

    def forward(self, client: Client, experiment_id: str, parameters: dict) -> Experiment:
        """
        Updates the parameters of an existing experiment in Picsellia.

        Args:
            client (Client): 
                An authenticated Picsellia client instance.
            experiment_id (str): 
                The unique ID of the experiment to update.
            parameters (dict): 
                A dictionary of parameters to update. Keys are parameter names, and values are the new parameter values.

        Returns:
            Experiment: The updated Experiment object.

        Raises:
            ValueError: If the experiment cannot be retrieved or updated due to an error.
        """
        try:
            # Retrieve the experiment by ID
            experiment = client.get_experiment_by_id(experiment_id)
            log = experiment.get_log('parameters')
            new_parameters = log.data | parameters
            # Update experiment parameters
            experiment.log_parameters(new_parameters)

            # Return the updated experiment
            return experiment

        except Exception as e:
            raise ValueError(f"Failed to update parameters for experiment with ID {experiment_id}: {str(e)}")


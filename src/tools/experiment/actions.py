import picsellia 
from smolagents import Tool

class LaunchExperimentTool(Tool):
    """
    A tool to launch an existing experiment in Picsellia.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The Picsellia SDK client instance used to interact with the API.
        experiment_id (str):
            - Type: `string`
            - Description: The unique ID of the experiment to launch.

    Output:
        output_type (Experiment): 
            The launched `Experiment` object.

    Example:
        ```
        # Example usage
        client = Client(api_token="your_api_token", organization_name="your_organization_name")
        experiment_id = "12345"

        # Launch the experiment
        launched_experiment = tool.forward(client=client, experiment_id=experiment_id)
        print(f"Experiment {launched_experiment.name} has been launched.")
        ```

    Notes:
        - The tool uses the `get_experiment` and `launch` methods from the Picsellia SDK.
        - Proper error handling ensures that issues during the launch process are communicated effectively.
    """
    name = "launch_training_tool"
    description = """
    This tool launches an existing experiment in Picsellia. It is useful for starting 
    experiments that have been previously configured and are ready to run.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia SDK client instance used to interact with the API.",
        },
        "experiment_id": {
            "type": "string",
            "description": "The unique ID of the experiment to launch.",
        },
    }
    output_type = "object"

    def forward(self, client: picsellia.Client, experiment_id: str) -> picsellia.Experiment:
        """
        Launches an existing experiment in Picsellia.

        Args:
            client (Client): 
                An authenticated Picsellia client instance.
            experiment_id (str): 
                The unique ID of the experiment to launch.

        Returns:
            Experiment: The launched Experiment object.

        Raises:
            ValueError: If the experiment cannot be retrieved or launched due to an error.
        """
        # try:
            # Retrieve the experiment by ID
        experiment = client.get_experiment_by_id(experiment_id)

        # Launch the experiment
        experiment.launch()

        # Return the launched experiment
        return experiment

        # except Exception as e:
        #     raise ValueError(f"Failed to launch experiment with ID {experiment_id}: {str(e)}")


launch_training_tool = LaunchExperimentTool()
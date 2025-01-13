from smolagents import Tool
import os
from picsellia import Client, DatasetVersion, Label, Project
from picsellia.types.schemas import InferenceType
from typing import List
import picsellia 

class ExperimentCreatorTool(Tool):
    name = "create_picsellia_experiment"
    description = """
    This tool create a Picsellia Experiment in a Picsellia Project  
    
    """
    inputs = {
        "project": {
            "type": "object",
            "description": "a Picsellia Project object",
        },
        "name": {
            "type": "string",
            "description": "name of the experiment to create",
        },
        "description": {
            "type": "string",
            "description": "description of the experiment, it should be as extensive as possible",
        },
    }
    output_type = "object"

    def forward(self, project: picsellia.Project, name: str, description: str) -> picsellia.Experiment:
        """
        Creates a new experiment in the specified Picsellia project.

        This method facilitates the creation of an experiment by leveraging the `create_experiment` 
        method provided by the Picsellia SDK. The experiment is associated with a specific project, 
        given a name, and description.

        Args:
            client (picsellia.Client):
                The Picsellia client instance used to interact with the Picsellia API.
            project (picsellia.Project):
                The Picsellia project object in which the experiment will be created.
            name (str):
                The name of the experiment to be created.
            description (str):
                A brief description of the experiment, explaining its purpose or scope.
            

        Returns:
            picsellia.Experiment:
                The newly created experiment object, which can be further used for configuration 
                or monitoring within the Picsellia platform.

        Example:
            ```
            # Example usage
            experiment = tool.forward(
                client=some_client_object,
                project=some_project_object,
                name="New Experiment",
                description="An experiment to test model performance on dataset X",
            )
            print(f"Created experiment: {experiment.name}")
            ```

        Raises:
            ValueError:
                If an error occurs during the experiment creation process, an appropriate error message 
                will be raised.

        Notes:
            - The method ensures that the created experiment is properly linked to the specified project.
        """
        experiment = project.create_experiment(name=name, description=description)
        return experiment
            
        # except Exception as e:
        #     raise ValueError(f"Failed to create labels: {str(e)}")


class AttachModelVersionToExperimentTool(Tool):
    """
    A tool to attach a specific ModelVersion to an existing experiment in Picsellia in order to train the ModelVersion.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The Picsellia SDK client instance used to interact with the API.
        experiment_id (str):
            - Type: `string`
            - Description: The unique ID of the experiment to which the model version will be attached.
        model_version (picsellia.ModelVersion):
            - Type: `picsellia.ModelVersion`
            - Description: The ModelVersion to attach to the experiment.

    Output:
        output_type (Experiment): 
            The updated `Experiment` object with the attached ModelVersion.

    Example:
        ```
        # Example usage
        client = Client(api_token="your_api_token", organization_name="your_organization_name")
        experiment_id = "12345"

        # Attach the model version to the experiment
        updated_experiment = tool.forward(client=client, experiment_id=experiment_id, model_version=model_version)
        print(f"Model version {model_version.id} attached to experiment {updated_experiment.name}.")
        ```

    Notes:
        - The tool uses the `get_experiment`, and `attach_model_version` methods from the Picsellia SDK.
        - Proper error handling ensures that issues during attachment are communicated effectively.
    """
    name = "attach_model_version_to_experiment"
    description = """
    This tool attaches a specific ModelVersion to an existing experiment in Picsellia. 
    It retrieves the experiment by its IDs, and link it to the ModelVersion.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia SDK client instance used to interact with the API.",
        },
        "experiment_id": {
            "type": "string",
            "description": "The unique ID of the experiment to which the model version will be attached.",
        },
        "model_version": {
            "type": "object",
            "description": "The ModelVersion to attach to the experiment.",
        },
    }
    output_type = "object"

    def forward(self, client: Client, experiment_id: str, model_version: picsellia.ModelVersion) -> picsellia.Experiment:
        """
        Attaches a specific ModelVersion to an existing experiment in Picsellia.

        Args:
            client (Client): 
                An authenticated Picsellia client instance.
            experiment_id (str): 
                The unique ID of the experiment.
            model_version (picsellia.ModelVersion): 
                The ModelVersion to attach.

        Returns:
            Experiment: The updated Experiment object with the attached ModelVersion.

        Raises:
            ValueError: If the ModelVersion or Experiment cannot be retrieved or updated due to an error.
        """
        # try:
            # Retrieve the experiment by ID
        experiment = client.get_experiment_by_id(experiment_id)

        # Attach the ModelVersion to the experiment
        experiment.attach_model_version(model_version)

        # Return the updated experiment
        return experiment

        # except Exception as e:
        #     raise ValueError(f"Failed to attach ModelVersion {model_version.id} to experiment {experiment_id}: {str(e)}")

class AttachDatasetVersionToExperimentTool(Tool):
    """
    
    """
    name = "attach_dataset_version_to_experiment"
    description = """
    This tool attaches a specific DatasetVersuib to an existing experiment in Picsellia. 
    It retrieves the DatasetVersion by its IDs, and link it to the Experiment.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia SDK client instance used to interact with the API.",
        },
        "experiment_id": {
            "type": "string",
            "description": "The unique ID of the experiment to which the model version will be attached.",
        },
        "dataset_version_id": {
            "type": "object",
            "description": "The DatasetVersion id of the DatasetVersion to attach to the experiment.",
        },

    }
    output_type = "object"

    def forward(self, client: Client, experiment_id: str, dataset_version_id: str) -> picsellia.Experiment:
        """
        Attaches a specific ModelVersion to an existing experiment in Picsellia.

        Args:
            client (Client): 
                An authenticated Picsellia client instance.
            experiment_id (str): 
                The unique ID of the experiment.
            dataset_version_id (str): 
                The DatasetVersion ID of the DatasetVersion to attach.

        Returns:
            Experiment: The updated Experiment object with the attached ModelVersion.

        Raises:
            ValueError: If the ModelVersion or Experiment cannot be retrieved or updated due to an error.
        """
        # try:
            # Retrieve the experiment by ID
        experiment = client.get_experiment_by_id(experiment_id)
        dataset_version = client.get_dataset_version_by_id(dataset_version_id)

        # Attach the ModelVersion to the experiment
        experiment.attach_dataset(name=dataset_version.version, dataset_version=dataset_version)

        # Return the updated experiment
        return experiment

        # except Exception as e:
        #     raise ValueError(f"Failed to attach ModelVersion {model_version.id} to experiment {experiment_id}: {str(e)}")



create_picsellia_experiment = ExperimentCreatorTool()
attach_model_version_to_experiment = AttachModelVersionToExperimentTool()
attach_dataset_to_experiment = AttachDatasetVersionToExperimentTool()
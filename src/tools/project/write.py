from smolagents import Tool
import os
from picsellia import Client, DatasetVersion, Label, Project, Experiment
from picsellia.types.schemas import InferenceType
from typing import List
import picsellia 
from picsellia.exceptions import ResourceConflictError

class ProjectCreatorTool(Tool):
    name = "get_or_create_picsellia_project"
    description = """
    This tool get or create a Picsellia Project  
    Methods:
    forward(name: str) -> picsellia.Project:
        - Functionality: Accepts a DatasetVersion and a name labels, creates the labels in the dataset version, and returns the created label objects.
        - Returns: A Picsellia `Project` objects.

    Use Case:
        - When creating a new model, you should create a project to encapsulate all the predictions.
        - BE CAREFUL - The project name need to be unique
    """
    inputs = {
        "project_name": {
            "type": "string",
            "description": "The name of the project to create",
        },
        "client": {
            "type": "object",
            "description": "Picsellia SDK client instance",
        }
    }
    output_type = "object"

    def forward(self, client: picsellia.Client, project_name: str) -> picsellia.Project:
        """
        - Returns: A Picsellia `Project` objects.
        Use Case:
        - When creating a new model, you should create a project to encapsulate all the predictions.
        - BE CAREFUL - The project name need to be unique       
        """
        try:
            project = client.create_project(name=project_name)
        except ResourceConflictError as e:
            project = client.get_project(project_name=project_name)
        return project
            
        # except Exception as e:
        #     raise ValueError(f"Failed to create labels: {str(e)}")



class AttachDatasetVersionToProjectTool(Tool):
    """
    A tool to attach a specific dataset version to a project in Picsellia.

    This tool facilitates linking a `DatasetVersion` object to a specific project, enabling
    seamless integration of datasets with project workflows in Picsellia.

    Attributes:
        name (str): The unique identifier for the tool, `"attach_dataset_version_to_project"`.
        description (str): A brief description of the tool's functionality.

    Inputs:
        project (Project):
            - Type: `object`
            - Description: The Picsellia project to which the dataset version will be attached.
        dataset_version (DatasetVersion):
            - Type: `object`
            - Description: The dataset version to attach to the specified project.

    Output:
        output_type (object): 
            A confirmation object or response indicating the success of the attachment operation.

    Example:
        ```
        # Example usage
        project = some_project_object
        dataset_version = some_dataset_version_object

        # Attach the dataset version to the project
        response = tool.forward(project, dataset_version)
        print(response)
        ```

    Notes:
        - The tool ensures proper error handling to manage issues during the attachment process.
        - It relies on Picsellia's SDK methods for attaching dataset versions to projects.
    """
    name = "attach_dataset_version_to_project"
    description = """
    This tool attaches a specific DatasetVersion to a Picsellia project. It facilitates linking 
    datasets to projects for seamless project management.
    """
    inputs = {
        "project": {
            "type": "object",
            "description": "The Picsellia project to which the dataset version will be attached.",
        },
        "dataset_version": {
            "type": "object",
            "description": "The dataset version to attach to the specified project.",
        },
    }
    output_type = "object"

    def forward(self, project: Project, dataset_version: DatasetVersion) -> object:
        try:
            # Attach the dataset version to the project
            project.attach_dataset(dataset_version)
            return {"status": "success", "message": f"Dataset version {dataset_version.version} attached to project {project.name}."}
        except Exception as e:
            raise ValueError(f"Failed to attach dataset version to project: {str(e)}")


class CreateExperimentFromPastExperimentTool(Tool):
    """
    Creates a new Picsellia experiment by cloning settings from a past experiment.

    This tool allows users to create a new experiment while inheriting configurations 
    and settings from an existing experiment, making it easier to iterate on previous work.

    Input:
        project (Project):
            - Type: `object` 
            - Description: The Picsellia project where the new experiment will be created.
        base_experiment (Experiment):
            - Type: `object`
            - Description: The source experiment to clone settings from.
        name (str):
            - Type: `string`
            - Description: Name for the new experiment.

    Output:
        output_type (object):
            A newly created Picsellia experiment object with settings inherited from the base experiment.

    Example:
        ```
        # Example usage
        project = some_project_object
        base_experiment = some_experiment_object
        name = "iteration-2"

        # Create new experiment from base experiment
        new_experiment = tool.forward(project, base_experiment, name)
        print(new_experiment)
        ```

    Notes:
        - The tool copies relevant settings like hyperparameters, model architecture, etc.
        - Inherits files, parameters, and labelmap from the base experiment
        - Dataset attachments and results are not copied over
        - Ensures proper error handling during experiment creation
    """
    name = "create_experiment_from_base_experiment"
    description = """
    Creates a new Picsellia experiment by cloning settings from a base experiment. Useful for 
    iterating on previous experiments while maintaining consistent configurations.
    """
    inputs = {
        "project": {
            "type": "object",
            "description": "The Picsellia project where the new experiment will be created.",
        },
        "base_experiment": {
            "type": "object", 
            "description": "The source experiment to clone settings from.",
        },
        "name": {
            "type": "string",
            "description": "Name for the new experiment.",
        }
    }
    output_type = "object"

    def forward(self, project: Project, base_experiment: Experiment, name: str) -> Experiment:
        try:
            # Create new experiment using base_experiment parameter
            new_experiment = project.create_experiment(
                name=name,
                base_experiment=base_experiment
            )
            return new_experiment
        except Exception as e:
            raise ValueError(f"Failed to create new experiment from base experiment: {str(e)}")



create_picsellia_project = ProjectCreatorTool()
attach_dataset_to_project = AttachDatasetVersionToProjectTool()

from smolagents import Tool
from picsellia import Client, Project, Experiment
from typing import List, Dict

class GetProjectByNameTool(Tool):
    """
    A tool to retrieve a specific project by its name from Picsellia.

    This tool searches for and returns a project matching the provided name
    in the Picsellia platform.

    Inputs:
        client (Client):
            - Type: `object` 
            - Description: The authenticated Picsellia client instance.
        project_name (str):
            - Type: `string`
            - Description: Name of the project to retrieve.

    Output:
        output_type (object):
            A Picsellia Project object representing the requested project.
            Returns None if no project is found with the given name.

    Example:
        ```python
        client = some_client_instance
        project = tool.forward(client, "my-project")
        if project:
            print(f"Found project: {project.name} with ID: {project.id}")
        else:
            print("Project not found")
        ```
    """
    name = "get_project_by_name"
    description = """
    This tool retrieves a specific project by name from Picsellia.
    Returns the Project object if found, None otherwise.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "project_name": {
            "type": "string",
            "description": "Name of the project to retrieve",
        }
    }
    output_type = "object"

    def forward(self, client: Client, project_name: str) -> Project:
        """
        Retrieves a project by its name.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            project_name (str): Name of the project to retrieve
            
        Returns:
            Project: Project object if found, None otherwise
            
        Raises:
            ValueError: If project retrieval fails
        """
        try:
            project = client.get_project(project_name=project_name)
            return project
        except Exception as e:
            raise ValueError(f"Failed to retrieve project: {str(e)}")


class ListProjectsTool(Tool):
    """
    A tool to list all accessible projects in Picsellia.

    This tool retrieves all projects that are accessible to the authenticated user
    in the Picsellia platform.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The authenticated Picsellia client instance.

    Output:
        output_type (list): 
            A list of Picsellia Project objects representing all accessible projects.

    Example:
        ```python
        client = some_client_instance
        projects = tool.forward(client)
        for project in projects:
            print(f"Project: {project.name}, ID: {project.id}")
        ```
    """
    name = "list_projects"
    description = """
    This tool lists all projects accessible to the authenticated user in Picsellia.
    It returns a list of Project objects containing project information.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        }
    }
    output_type = "object"

    def forward(self, client: Client) -> List[Project]:
        """
        Lists all projects accessible to the authenticated user.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            
        Returns:
            List[Project]: List of Project objects
        """
        try:
            return client.projects.list()
        except Exception as e:
            raise ValueError(f"Failed to list projects: {str(e)}")

list_projects = ListProjectsTool()

class ListExperimentsTool(Tool):
    """
    This tool retrieves all experiments within a specified project in the Picsellia platform.

    Inputs:
        client (Client):
            - Type: `object` 
            - Description: The authenticated Picsellia client instance.
        project (Project):
            - Type: `object`
            - Description: The Picsellia Project object to list experiments from.

    Output:
        output_type (list):
            A list of Picsellia Experiment objects representing all experiments in the project.

    Example:
        ```python
        client = some_client_instance
        project = some_project_instance
        experiments = tool.forward(client, project)
        for experiment in experiments:
            print(f"Experiment: {experiment.name}, ID: {experiment.id}")
        ```
    """
    name = "list_experiments"
    description = """
    This tool lists all experiments within a specified Picsellia project.
    It returns a list of Experiment objects containing experiment information.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "project_id": {
            "type": "string", 
            "description": "Picsellia Project ID",
        }
    }
    output_type = "object"

    def forward(self, client: Client, project_id: str) -> List[Experiment]:
        """
        Lists all experiments in the specified project.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            project_id (str): Project ID to list experiments from
            
        Returns:
            List[Experiment]: List of Experiment objects
        """
        try:
            return client.get_project_by_id(project_id).list_experiments()
        except Exception as e:
            raise ValueError(f"Failed to list experiments: {str(e)}")

list_experiments = ListExperimentsTool()

class ListExperimentAttachmentsAndLogsTool(Tool):
    """Tool for listing all attached objects and logs in a Picsellia Experiment.

    This tool retrieves all attachments (datasets, models, files etc.) and logs 
    associated with a given experiment.

    Input:
        client (Client):
            - Type: Picsellia Client object
            - Description: Authenticated Picsellia client instance
        experiment (Experiment): 
            - Type: Picsellia Experiment object
            - Description: The Picsellia Experiment to list attachments and logs from

    Output:
        output_type (dict):
            A dictionary containing:
            - attachments: Dict with lists of different attached objects
                - datasets: List of attached dataset versions
                - model_files: List of stored model files/weights
                - other_files: List of other attached files
            - logs: Dict of experiment logs by name
                - metrics: Training metrics logs
                - parameters: Experiment parameters
                - labelmap: Model label mapping if exists
                - other_logs: Additional experiment logs

    Example:
        ```python
        client = some_client_instance
        experiment = some_experiment_instance
        results = tool.forward(client, experiment)
        attachments = results['attachments']
        logs = results['logs']
        ```
    """
    name = "list_experiment_attachments_and_logs"
    description = """
    This tool lists all attachments and logs within a specified Picsellia experiment including datasets,
    model files, stored artifacts, training metrics, parameters and other logs.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "experiment_id": {
            "type": "string", 
            "description": "Picsellia Experiment ID to get attachments and logs from",
        }
    }
    output_type = "object"

    def forward(self, client: Client, experiment_id: str) -> Dict[str, Dict]:
        """
        Lists all attachments and logs in the specified experiment.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            experiment (str): Experiment ID
            
        Returns:
            Dict[str, Dict]: Dictionary containing:
                attachments (Dict):
                    - datasets (List[DatasetVersion]): List of attached dataset versions
                    - model_files (List[File]): List of model files/weights from base model
                    - other_files (List[File]): List of other stored files/artifacts
                logs (Dict):
                    - metrics (Dict[str, Any]): Training metrics like accuracy, loss etc.
                        Contains data for 'accuracy', 'loss', 'train-split', 'test-split', 'eval-split'
                    - parameters (Dict[str, Any]): Experiment hyperparameters and settings
                    - labelmap (Dict[str, Any]): Model label mapping configuration
                    - other_logs (Dict[str, Any]): Any additional experiment logs
        """
        try:
            # Get experiment by ID
            experiment = client.get_experiment_by_id(experiment_id)
            # Get all attachments
            attachments = {
                'datasets': experiment.list_attached_dataset_versions(),
                'model_files': [],
                'other_files': []
            }
            
            # Get base model files if they exist
            try:
                base_model = experiment.get_base_model_version()
                if base_model:
                    attachments['model_files'].extend(base_model.list_files())
            except:
                pass
                
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
                'labelmap': {},
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
                
            return {
                'attachments': attachments,
                'logs': logs
            }
            
        except Exception as e:
            raise ValueError(f"Failed to list experiment attachments and logs: {str(e)}")

list_experiment_attachments_and_logs = ListExperimentAttachmentsAndLogsTool()



class GetExperimentTool(Tool):
    """Tool for retrieving a specific experiment from a project by name or ID.

    This tool retrieves a single experiment from a specified project using either 
    the experiment name or ID.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The authenticated Picsellia client instance.
        project (Project):
            - Type: `object`
            - Description: The Picsellia Project object to search in.
        experiment_identifier (str):
            - Type: `str`
            - Description: Name or ID of the experiment to retrieve.

    Output:
        output_type (object):
            A Picsellia Experiment object representing the requested experiment.

    Example:
        ```python
        client = some_client_instance
        project = some_project_instance
        experiment = tool.forward(client, project, "my-experiment")
        print(f"Found experiment: {experiment.name} (ID: {experiment.id})")
        ```
    """
    name = "get_experiment"
    description = """
    This tool retrieves a specific experiment from a project using its name or ID.
    It returns the Experiment object if found.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "project": {
            "type": "object",
            "description": "Picsellia Project object",
        },
        "experiment_identifier": {
            "type": "string",
            "description": "Name or ID of the experiment to retrieve",
        }
    }
    output_type = "object"

    def forward(self, client: Client, project: Project, experiment_identifier: str) -> Experiment:
        """
        Retrieves a specific experiment from the project.
        
        Args:
            client (Client): Authenticated Picsellia client instance
            project (Project): Project to search in
            experiment_identifier (str): Name or ID of the experiment
            
        Returns:
            Experiment: The requested experiment object
            
        Raises:
            ValueError: If experiment cannot be found or other errors occur
        """
        try:
            # First try to get by ID
            try:
                return client.get_experiment_by_id(experiment_identifier)
            except:
                pass
            
            # If ID lookup fails, search by name in project experiments
            experiments = project.list_experiments()
            for experiment in experiments:
                if experiment.name == experiment_identifier:
                    return experiment
                    
            raise ValueError(f"No experiment found with identifier '{experiment_identifier}'")
            
        except Exception as e:
            raise ValueError(f"Failed to get experiment: {str(e)}")

get_experiment = GetExperimentTool()
get_project_by_name = GetProjectByNameTool()
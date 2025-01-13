import picsellia 
from smolagents import Tool
from picsellia import DatasetVersion, Asset, Label, Client
from picsellia.types.schemas import DatasetVersionStats
from typing import List, Dict

class ListDatasetVersionAssetsTool(Tool):
    name = "list_dataset_version_assets"
    description = """
    This tool lists all assets within a given DatasetVersion.
    """
    inputs = {
        "dataset_version": {
            "type": "object",
            "description": "The DatasetVersion object from which to list assets",
        },
    }
    output_type = "any"

    def forward(self, dataset_version: DatasetVersion) -> List[Asset]:
        try:
            # Get all assets from the dataset version
            assets = dataset_version.list_assets()
            return assets
            
        except Exception as e:
            raise ValueError(f"Failed to list assets from dataset version: {str(e)}")



class ListDatasetVersionLabelsTool(Tool):
    name = "list_dataset_version_labels"
    description = """
    This tool lists all unique labels within a given DatasetVersion.
    Returns a list of Picsellia Label objects
    """
    inputs = {
        "dataset_version": {
            "type": "object",
            "description": "The DatasetVersion object from which to list labels",
        },
    }
    output_type = "any"

    def forward(self, dataset_version: DatasetVersion) -> List[Label]:
        """ 
        Lists all unique labels in the specified DatasetVersion.

        This method fetches all labels associated with the provided DatasetVersion object.

        Args:
            dataset_version (DatasetVersion): The DatasetVersion object from which to retrieve labels.


        Returns:
            List[Label]: A list of picsellia.Label

        Raises:
            ValueError: If the labels cannot be retrieved due to an error.
        """
        try:
            # Get all labels from the dataset version
            labels = dataset_version.list_labels()
            return labels
            
        except Exception as e:
            raise ValueError(f"Failed to list labels from dataset version: {str(e)}")


class LabelExistenceChecker(Tool):
    name = "check_if_label_exists"
    description = """
    This tool check if a `Label`object exists for a given Label Name.
    Returns True or False
    """
    inputs = {
        "dataset_version": {
            "type": "object",
            "description": "The DatasetVersion object from which to list labels",
        },
        "label_name": {
            "type": "string",
            "description": "Label name to check for existence"
        }
    }
    output_type = "boolean"

    def forward(self, dataset_version: DatasetVersion, label_name: str) -> bool:
        """ 
        Lists all unique labels in the specified DatasetVersion.

        This method check the existence of a given Label in DatasetVersion.

        Args:
            dataset_version (DatasetVersion): The DatasetVersion object from which to retrieve labels.
            label_name (str): Label name to check.

        Returns:
            bool: True if label exists

        """
            # Get all labels from the dataset version
        labels = dataset_version.list_labels()
        if len(labels) == 0: return False
        for label in labels:
            if label.name == label_name:
                return True
        return False
            


class DatasetVersionObjectRepartitionTool(Tool):
    """
    A tool to fetch and display statistics of a specified DatasetVersion in Picsellia.

    This tool retrieves comprehensive statistics for a given DatasetVersion, including:
    - Object distribution across different labels.
    - Total number of objects.
    - Number of annotated images.

    Attributes:
        name (str): The unique name identifier for the tool.
        description (str): A brief description of the tool's functionality.
        inputs (dict): A dictionary defining the expected input parameters, their types, and descriptions.
        output_type (str): The type of output the tool returns; in this case, an object containing dataset statistics.

    Methods:
        forward(dataset_version_id: str) -> picsellia.types.schemas.DatasetVersionStats:
            Retrieves and returns the statistics for the specified DatasetVersion.
    """

    name = "dataset_version_repartition_viewer"
    description = """
    This tool fetches all the statistics of a DatasetVersion in Picsellia, providing the object distribution,
    the total number of objects, and the number of annotated images.
    """
    inputs = {
        "dataset_version": {
            "type": "object",
            "description": "The Picsellia DatasetVersion object for which to retrieve statistics.",
        },
    }
    output_type = "object"

    def forward(self, dataset_version: DatasetVersion) -> picsellia.types.schemas.DatasetVersionStats:
        """
        Retrieves statistics for the specified DatasetVersion.

        This method connects to the Picsellia platform, fetches the DatasetVersion corresponding to the provided ID,
        and retrieves its statistics, including object distribution, total object count, and annotated image count.

        Args:
            dataset_version_id (str): The unique identifier of the DatasetVersion in Picsellia.

        Returns:
            picsellia.types.schemas.DatasetVersionStats: An object containing the statistics of the DatasetVersion.

        Raises:
            picsellia.exceptions.ResourceNotFoundError: If the DatasetVersion with the specified ID does not exist.
            picsellia.exceptions.APIError: If there is an issue communicating with the Picsellia API.

        Example:
            >>> tool = DatasetVersionObjectRepartitionTool()
            >>> stats = tool.forward("123e4567-e89b-12d3-a456-426614174000")
            >>> print(stats.object_distribution)
        """
        # Retrieve the DatasetVersion object from Picsellia using its ID

        # Fetch and return the statistics for the retrieved DatasetVersion
        return dataset_version.retrieve_stats()
    
class FetchDatasetVersionByIDTool(Tool):
    name = "fetch_dataset_version_by_id"
    description = """
    This tool fetches a DatasetVersion object from Picsellia using its ID.
    It requires an initialized Picsellia Client and the dataset version ID.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "An initialized Picsellia Client instance",
        },
        "dataset_version_id": {
            "type": "string",
            "description": "The ID of the dataset version to fetch",
        }
    }
    output_type = "object"

    def forward(self, client: Client, dataset_version_id: str) -> DatasetVersion:
        try:
            dataset_version = client.get_dataset_version_by_id(id=dataset_version_id)
            return dataset_version
        except Exception as e:
            raise ValueError(f"Failed to fetch dataset version with ID {dataset_version_id}: {str(e)}")


class FetchDatasetVersionByNameAndVersionTool(Tool):
    name = "fetch_dataset_version_by_name_and_version"
    description = """
    This tool fetches a DatasetVersion object from Picsellia using its name and version.
    It requires an initialized Picsellia Client and the dataset name and version.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "An initialized Picsellia Client instance",
        },
        "dataset_name": {
            "type": "string",
            "description": "The name of the dataset version to fetch",
        },
        "dataset_version": {
            "type": "string",
            "description": "The version of the dataset to fetch"
        }
    }
    output_type = "object"

    def forward(self, client: Client, dataset_name: str, dataset_version: str) -> DatasetVersion:
        try:
            dataset_version = client.get_dataset(name=dataset_name).get_version(dataset_version)
            return dataset_version
        except Exception as e:
            raise ValueError(f"Failed to fetch dataset version with name {dataset_name}/{dataset_version}: {str(e)}")


fetch_dataset_version_by_name_and_version = FetchDatasetVersionByNameAndVersionTool()
fetch_dataset_version_by_id_tool = FetchDatasetVersionByIDTool()
list_dataset_assets_tool = ListDatasetVersionAssetsTool()
list_dataset_labels_tool = ListDatasetVersionLabelsTool()
dataset_version_repartition_viewer = DatasetVersionObjectRepartitionTool()
check_if_label_exists = LabelExistenceChecker()
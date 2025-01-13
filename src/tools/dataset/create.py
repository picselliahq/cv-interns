from smolagents import Tool
import os
from picsellia import Client, DatasetVersion, Label
from picsellia.types.schemas import InferenceType
from typing import List, Tuple
from random import shuffle

class LabelCreatorTool(Tool):
    name = "create_picsellia_label_object"
    description = """
    This tool create Label Picselli object in a DatasetVersion of Picsellia 
    Methods:
    forward(dataset_version: DatasetVersion, label: str) -> Label:
        - Functionality: Accepts a DatasetVersion and a name labels, creates the labels in the dataset version, and returns the created label objects.
        - Returns: A Picsellia `Label` objects.
        - Exceptions: Raises a `ValueError` if the label creation fails, providing an error message with details.

    Use Case:
        This tool is particularly useful in scenarios where new labels need to be added to a specific version of a dataset in Picsellia. It ensures that the process of label creation is streamlined and error-handled effectively.
    """
    inputs = {
        "dataset_version": {
            "type": "object",
            "description": "The DatasetVersion object to check and update",
        },
        "label": {
            "type": "string",
            "description": "The name of labels to create, it can be only one",
        }
    }
    output_type = "object"

    def forward(self, dataset_version: DatasetVersion, label: str) -> List[Label]:
        """
        Creates label objects within a specified Picsellia DatasetVersion.

        This method accepts a dataset version and a list of label names to create. It ensures that the labels
        are added to the dataset version without duplication, and returns the created label objects.

        Args:
            dataset_version (DatasetVersion):
                The specific version of the dataset in which the labels will be created. This is a Picsellia
                object that manages dataset version details.
            labels (List[str]):
                A list of label names (strings) to be created in the dataset version. The list can contain
                one or more label names.

        Returns:
            Label:
                A Picsellia `Label` object that was successfully created in the dataset version.

        Raises:
            ValueError:
                If an error occurs during label creation, a `ValueError` is raised with a detailed error
                message.            
        """
        # try:
            # Get current inference type
        labels = dataset_version.create_label(name=label)
        return labels
            
        # except Exception as e:
        #     raise ValueError(f"Failed to create labels: {str(e)}")


class SplitDatasetVersionInTrainTestValDatasetVersionsTool(Tool):
    """
    Splits a DatasetVersion in Picsellia into three separate DatasetVersions for training, testing, 
    and validation based on specified ratios.

    This tool takes an existing DatasetVersion, shuffles its assets, and splits them into three 
    distinct sets according to the provided ratios. It then creates new DatasetVersions for each 
    subset: one for training, one for testing, and one for validation.
    """
    name = "create_train_test_val_dataset_version"
    description = """
    This tool splits a DatasetVersion in Picsellia into three separate DatasetVersions for 
    training, testing, and validation, based on the specified ratios.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia SDK client instance used to interact with the API.",
        },
        "dataset_version_id": {
            "type": "string",
            "description": "The unique ID of the DatasetVersion to split.",
        },
        "train_ratio": {
            "type": "number",
            "description": "The proportion of data to allocate to the training set (e.g., 0.7 for 70%).",
        },
        "test_ratio": {
            "type": "number",
            "description": "The proportion of data to allocate to the test set (e.g., 0.2 for 20%).",
        },
        "val_ratio": {
            "type": "number",
            "description": "The proportion of data to allocate to the validation set (e.g., 0.1 for 10%).",
        },
    }
    output_type = "object"

    def forward(
        self,
        client: Client,
        dataset_version_id: str,
        train_ratio: float,
        test_ratio: float,
        val_ratio: float,
    ) -> Tuple[DatasetVersion, DatasetVersion, DatasetVersion]:
        """
        Splits a DatasetVersion into training, testing, and validation subsets and creates separate 
        DatasetVersions for each subset.

        This method takes an existing DatasetVersion, shuffles its assets, and splits them into three 
        distinct subsets based on the specified ratios. It then creates new DatasetVersions for each subset 
        and returns them.

        Args:
            client (Client):
                An authenticated Picsellia client instance used to interact with the API.
            dataset_version_id (str):
                The unique ID of the DatasetVersion to be split.
            train_ratio (float):
                The proportion of the data to allocate to the training subset (e.g., 0.7 for 70%).
            test_ratio (float):
                The proportion of the data to allocate to the testing subset (e.g., 0.2 for 20%).
            val_ratio (float):
                The proportion of the data to allocate to the validation subset (e.g., 0.1 for 10%).

        Returns:
            Tuple[DatasetVersion, DatasetVersion, DatasetVersion]:
                A tuple containing three new DatasetVersion objects:
                - The first DatasetVersion is for training.
                - The second DatasetVersion is for testing.
                - The third DatasetVersion is for validation.

        Raises:
            ValueError:
                - If the sum of `train_ratio`, `test_ratio`, and `val_ratio` is not equal to 1.0.
                - If any error occurs during the process of splitting or creating the new DatasetVersions.

        Example:
            ```python
            client = Client(api_token="your_api_token", organization_name="your_organization_name")
            dataset_version_id = "12345"
            train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1

            train, test, val = tool.forward(
                client=client,
                dataset_version_id=dataset_version_id,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                val_ratio=val_ratio
            )

            print(f"Training DatasetVersion: {train}")
            print(f"Testing DatasetVersion: {test}")
            print(f"Validation DatasetVersion: {val}")
            ```

        Notes:
            - The `train_ratio`, `test_ratio`, and `val_ratio` must sum to 1.0. If they do not, the method 
            raises a `ValueError`.
            - The assets in the DatasetVersion are shuffled before splitting to ensure randomness.
            - The new DatasetVersions inherit the type, labels, tags, and annotations of the original DatasetVersion.
        """
        # try:
            # Ensure the ratios sum to 1.0
        if round(train_ratio + test_ratio + val_ratio, 5) != round(1.0, 5):
            raise ValueError("The sum of train_ratio, test_ratio, and val_ratio must equal 1.0.")

        # Retrieve the dataset version by ID
        dataset_version = client.get_dataset_version_by_id(dataset_version_id)
        # annotation_path = dataset_version.export_annotation_file("COCO")
        assets = dataset_version.list_assets()
        shuffle(assets)
        train_end = int(len(assets) * train_ratio)
        val_end = train_end + int(len(assets) * val_ratio)
        train_assets = assets[:train_end]
        val_assets = assets[train_end:val_end]
        test_assets = assets[val_end:]

        try:
            train_dataset_version, _ = dataset_version.fork(version="train", assets=train_assets, type=dataset_version.type,
                                                            with_labels=True, with_tags=True, with_annotations=True)
        except Exception as e:
            
            train_dataset_version = client.get_dataset_by_id(dataset_version.origin_id).get_version('train')
        
        try:
            test_dataset_version, _ = dataset_version.fork(version="test", assets=test_assets, type=dataset_version.type,
                                                            with_labels=True, with_tags=True, with_annotations=True)
        except Exception as e:            
            test_dataset_version = client.get_dataset_by_id(dataset_version.origin_id).get_version('test')
        
        try:
            val_dataset_version, _ = dataset_version.fork(version="val", assets=val_assets, type=dataset_version.type,
                                                            with_labels=True, with_tags=True, with_annotations=True)
        except Exception as e:
            val_dataset_version = client.get_dataset_by_id(dataset_version.origin_id).get_version('val')
        

        return (train_dataset_version, test_dataset_version, val_dataset_version)
            

        # except Exception as e:
        #     raise ValueError(f"Failed to split DatasetVersion with ID {dataset_version_id}: {str(e)}")


create_picsellia_label_object = LabelCreatorTool()
create_train_test_val_dataset_version = SplitDatasetVersionInTrainTestValDatasetVersionsTool()


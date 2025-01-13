from smolagents import Tool
from picsellia import Datalake, Tag, DatasetVersion, Client
from picsellia.sdk.data import MultiData
from typing import List, Union


class CreateDatasetAndDatasetVersionTool(Tool):
    name = "create_dataset_and_first_version"
    description = """
    
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "a Picsellia SDK Client object"
        },
        "data": {
            "type": "object",
            "description": "a Picsellia MultiData object",
        },
        "name": {
            "type": "string",
            "description": "the name of the Dataset to create",
        },
    }
    output_type = "object"

    def forward(self, client: Client, data: MultiData, name: str) -> DatasetVersion:
        """
        
        """
        # try:
        dataset = client.create_dataset(name=name, private=True)
        dataset_version = dataset.create_version("initial")
        job = dataset_version.add_data(data=data, tags=["agent-added"])
        job.wait_for_done()
        return dataset_version
        # except Exception as e:
        #     raise ValueError(f"Failed to create dataset")


create_dataset_and_version_tool = CreateDatasetAndDatasetVersionTool()
from smolagents import Tool, ToolCallingAgent, HfApiModel, CodeAgent, ManagedAgent, GradioUI
from picsellia import Datalake, Tag, DatasetVersion, Client
from picsellia.sdk.data import MultiData
from typing import List, Union
import os

class CreateDatasetFromDataTagSearch(Tool):
    name = "create_dataset_from_data_search_by_tags"
    description = """
    Creates a dataset version from data items in the Picsellia Datalake that match the specified tags and adds a specific tag to the dataset version.
        Parameters:
            dataset_name (str): The name of the dataset to create.
            tags (List[str]): A list of tags used to filter the search results. Tags can be provided as strings.
            limit (int, optional): The maximum number of data items to fetch in the search. Defaults to 1000.

        Returns:
            str: The ID of the created dataset version.

        Raises:
            ValueError: If the search fails or if there is an error during dataset creation.
    """
    inputs = {
        "dataset_name": {
            "type": "string",
            "description": "name of the dataset to create"
        },
        "tags": {
            "type": "object",
            "description": "A list of tags to filter data by.",
        },
        "limit": {
            "type": "integer",
            "description": "The amount of data to fetch in a Datalake search", 
            "nullable": "True"
        },
    }
    output_type = "object"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Client(api_token=os.environ["PICSELLIA_TOKEN"])
        self.datalake = self.client.get_datalake(name='default')

    def forward(self, dataset_name: str, tags: List[str], limit: int = 1000) -> str:
        """
        Creates a dataset version from data items in the Picsellia Datalake that match the specified tags and adds a specific tag to the dataset version.

        Parameters:
            dataset_name (str): The name of the dataset to create.
            tags (List[str]): A list of tags used to filter the search results. Tags can be provided as strings.
            limit (int, optional): The maximum number of data items to fetch in the search. Defaults to 1000.

        Returns:
            str: The ID of the created dataset version.

        Raises:
            ValueError: If the search fails or if there is an error during dataset creation.
        """
        try:
            # Ensure tags are in the correct format
            formatted_tags = [self.datalake.get_or_create_data_tag(tag) if isinstance(tag, str) else tag for tag in tags]
            # Search for data in the Datalake with specific tags
            data_items = self.datalake.list_data(tags=formatted_tags, intersect_tags=True, limit=limit)
            dataset_version = self.client.create_dataset(dataset_name).create_version('agent-created')
            dataset_version.add_data(data=data_items, tags=['AI-ACTIONS'])
            return str(dataset_version.id)
        
        except Exception as e:
            raise ValueError(f"Failed to create dataset from tags {tags}: {str(e)}")



# create_dataset_from_data_search_by_tags = CreateDatasetFromDataTagSearch()

model_name = "meta-llama/Llama-3.3-70B-Instruct"
model = HfApiModel(model_name, token=os.getenv("HF_TOKEN"))

dataset_creator_agent = CodeAgent(
    model=model,
    tools=[CreateDatasetFromDataTagSearch()],

)

GradioUI(dataset_creator_agent).launch()
# dataset_creator_agent.run("create a dataset named stupid-test with data matching tags `dummy-tag`")

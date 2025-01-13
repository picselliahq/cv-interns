from smolagents import Tool
from picsellia import Datalake, Tag, Client
from picsellia.sdk.data import MultiData
from typing import List, Union


class SearchDataWithTagTool(Tool):
    name = "search_data_with_tag"
    description = """
    This tool searches for data in the Picsellia Datalake that matches specific tags.
    """
    inputs = {
        "datalake": {
            "type": "object",
            "description": "The Picsellia Datalake object to search in.",
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

    def forward(self, datalake: Datalake, tags: List[Union[str, Tag]], limit: int = 1000) -> MultiData:
        """
        Searches the Picsellia Datalake for data items that match the specified tags.

        Parameters:
        datalake (Datalake): The Datalake object to perform the search within.
        tags (List[Union[str, 'Tag']]): A list of tags used to filter the search results. Tags can be provided as 
        strings or as Tag objects.

        Returns:
        MultiData: A MultiData object that matches the specified tags.

        Raises:
        ValueError: If the search fails or an error occurs during tag processing or data retrieval.

        Example:
        --------
        >>> datalake = Datalake()
        >>> tags = ["cat", "dog"]
        >>> tool = SearchDataWithTagTool()
        >>> results = tool.forward(datalake, tags)
        >>> print(results)
        """
        try:
            # Ensure tags are in the correct format
            formatted_tags = [datalake.get_or_create_data_tag(tag) if isinstance(tag, str) else tag for tag in tags]
            
            # Search for data in the Datalake with specific tags
            data_items = datalake.list_data(tags=formatted_tags, intersect_tags=True, limit=limit)
            return data_items
        except Exception as e:
            raise ValueError(f"Failed to search for data with tags {tags}: {str(e)}")


class ListDatasetAndVersionTool(Tool):
    name = "list_dataset_and_dataset_version"
    description = """
    This tool searches for all the datasets and dataset versions in Picsellia and provide some metadata.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia Client object",
        },
    }
    output_type = "object"

    def forward(self, client: Client) -> object:
        """
        Generates a comprehensive report of all datasets and their versions within the Picsellia client.

        This method retrieves all datasets associated with the provided Picsellia `Client` object, iterates 
        through each dataset and its versions, and compiles metadata and version statistics into a structured 
        report.

        Args:
            client (Client):
                A Picsellia `Client` object used to access and retrieve datasets, their versions, and metadata.

        Returns:
            object:
                A dictionary containing a report of datasets and their versions. The report structure is:
                ```
                {
                    "dataset_name_1": [
                        {"version": version_number, 
                        "dataset_version_id": uuid, 
                        "metadata": {
                            "label_repartition": {
                                "label_1": count_1,
                                "label_2": count_2,
                                ...
                            },
                            "nb_objects": total_number_of_objects,
                            "nb_annotations": total_number_of_annotations
                    }},
                        ...
                    ],
                    "dataset_name_2": [
                        {"version": version_number, 
                        "dataset_version_id": uuid, 
                        "metadata": {
                            "label_repartition": {
                                "label_1": count_1,
                                "label_2": count_2,
                                ...
                            },
                            "nb_objects": total_number_of_objects,
                            "nb_annotations": total_number_of_annotations
                    }},
                    ],
                    ...
                }
                ```
                - `dataset_name`: The name of the dataset.
                - `version`: The version number of the dataset version.
                - `metadata`: Statistical metadata retrieved for the dataset version.

        Example:
            ```
            # Example usage
            client = some_picsellia_client_object

            # Generate dataset report
            report = tool.forward(client)
            print(report)
            ```

        Notes:
            - The method utilizes the `list_datasets` function of the `Client` object to retrieve datasets.
            - Each dataset's versions are retrieved using the `list_versions` function, and their stats are 
            fetched via `retrieve_stats`.
            - The output is a structured dictionary report for easy access and analysis of dataset details.
        """
        
        # Ensure tags are in the correct format
        report = {}
        datasets = client.list_datasets()
        for dataset in datasets:
            report[dataset.name] = [{'version': e.version, 'dataset_version_id': str(e.id), 'metadata': e.retrieve_stats()} for e in dataset.list_versions()]

        # Search for data in the Datalake with specific tags
        return report


list_data_in_datalake_through_tags = SearchDataWithTagTool()
list_all_datasets_and_dataset_versions_tool = ListDatasetAndVersionTool()
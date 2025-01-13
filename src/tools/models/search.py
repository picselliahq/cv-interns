from picsellia import Client
from picsellia import Model, ModelVersion
from picsellia.types.enums import InferenceType
from smolagents import Tool
from typing import List, Tuple


# class PublicModelLister(Tool):
#     name = "list_available_public_models_for_task"

#     description = """
#         This tool retrieves all public models and their versions available in Picsellia and filters them based on the specified task type. It is useful for identifying models suited for a particular type of inference, such as "OBJECT_DETECTION".

#         ### Inputs:
#         - `client`:
#         - **Type**: `object`
#         - **Description**: A Picsellia SDK client instance used to interact with the Picsellia API.

#         - `type`:
#         - **Type**: `string`
#         - **Description**: A string representing the desired inference type. It must be one of the predefined values in Picsellia's `InferenceType`.

#         ### Output:
#         - **Type**: `object`
#         - **Description**: A dictionary containing the names of public models as keys, with a list of their versions as values. Each version includes:
#         - `version`: The name of the model version.
#         - `type`: The inference type of the version.
#         - `model_version`: The model version object.

#         ### Example Usage:
#         ```python
#         # Example usage
#         client = Client(api_token="your_api_token", organization_name="your_organization_name")
#         inference_type = "OBJECT_DETECTION"

#         # Retrieve public models for the specified task
#         available_models = tool.forward(client=client, type=inference_type)
#         print(available_models)
#         Notes:
#         Only model versions matching the specified task type are included in the results.
#         If no versions match the given type for a model, that model will be excluded from the output. """
#     inputs = {
#         "client": {
#             "type": "object",
#             "description": "Picsellia SDK client instance",
#         },
#         "type": {
#             "type": "string",
#             "description": f"can be one of these values {InferenceType.values()}"
#         }
#     }
#     output_type = "object"

#     def forward(self, client: Client, type: str) -> dict:
#         f"""
#         Lists all public models and model versions in Picsellia and filters those
#         that are of type `type`.

#         InferenceType can be: {InferenceType.values()}

#         Args:
#             client (Client): 
#                 An authenticated Picsellia client instance.

#         Returns:
#             dict:
#                 A dictionary containing filtered public models and model versions with type "OBJECT_DETECTION".
#                 The structure is:
#                 {
#                    {
#                         'Model A name': [
#                             {'model_version': 'Version 1', 'type': 'OBJECT_DETECTION'}
#                         ],
#                         'Model B name': [
#                             {'model_version': 'Version 1', 'type': 'OBJECT_DETECTION'}
#                         ]
#                     }
#                 }

#         Example:
#             ```
#             # List object detection models
#             object_detection_models = list_public_object_detection_models(client)
#             print(object_detection_models)
#             ```

#         Notes:
#             - This function fetches public models and their versions from Picsellia.
#         """
#         available_models = {}

#         # try:
#             # Fetch all public models
#         public_models = client.list_public_models()

#         for model in public_models:
#             # Initialize the list of versions for the model
#             available_models[model.name] = []

#             # Fetch all model versions
#             for version in model.list_versions():
#                 if version.type.value == type:
#                     available_models[model.name].append({
#                         "version": version.name,
#                         "type": version.type
#                     })

#         # Remove models with no matching versions
#         # available_models = {k: v for k, v in available_models.items() if v}

#         return available_models

#         # except Exception as e:
#         #     raise ValueError(f"Failed to list object detection models: {str(e)}")

# class PublicModelLister(Tool):
#     name = "list_available_public_models_for_task"

#     description = """
#         This tool retrieves all public models and their versions available in Picsellia and filters them based on the specified task type. It is useful for identifying models suited for a particular type of inference, such as "OBJECT_DETECTION".

#         ### Inputs:
#         - `client`:
#         - **Type**: `object`
#         - **Description**: A Picsellia SDK client instance used to interact with the Picsellia API.

#         - `type`:
#         - **Type**: `string`
#         - **Description**: A string representing the desired inference type. It must be one of the predefined values in Picsellia's `InferenceType`.

#         ### Output:
#         - **Type**: `object`
#         - **Description**: A dictionary containing the names of public models as keys, with a list of their versions as values. Each version includes:
#         - `version`: The name of the model version.
#         - `type`: The inference type of the version.
#         - `model_version`: The model version object.

#         ### Example Usage:
#         ```python
#         # Example usage
#         client = Client(api_token="your_api_token", organization_name="your_organization_name")
#         inference_type = "OBJECT_DETECTION"

#         # Retrieve public models for the specified task
#         available_models = tool.forward(client=client, type=inference_type)
#         print(available_models)
#         Notes:
#         Only model versions matching the specified task type are included in the results.
#         If no versions match the given type for a model, that model will be excluded from the output. """
#     inputs = {
#         "client": {
#             "type": "object",
#             "description": "Picsellia SDK client instance",
#         },
#         "type": {
#             "type": "string",
#             "description": f"can be one of these values {InferenceType.values()}"
#         }
#     }
#     output_type = "object"

#     def forward(self, client: Client, type: str) -> List[Tuple[str, str, str]]:
#         f"""
#         Retrieves and filters public models and their versions in Picsellia based on a specified inference type.

#         This function fetches all public models available through the Picsellia client, iterates through their 
#         versions, and filters the results to include only those versions that match the specified inference type. 
#         The filtered results are returned as a list of tuples containing model details.

#         Args:
#             client (Client):
#                 An authenticated Picsellia client instance used to interact with the API.
#             type (str):
#                 The inference type to filter model versions by (e.g., "OBJECT_DETECTION", "CLASSIFICATION"). 
#                 Only model versions of this type will be included in the results.

#         Returns:
#             List[Tuple[str, str, str]]:
#                 A list of tuples, where each tuple contains:
#                 - `model.name` (str): The name of the model.
#                 - `version.version` (str): The version name of the model.
#                 - `version.id` (str): The unique ID of the model version.

#         Example:
#             ```python
#             # Example usage
#             client = Client(api_token="your_api_token", organization_name="your_organization_name")
#             inference_type = {InferenceType.values()}

#             # Retrieve and filter public models
#             available_models = tool.forward(client=client, type=inference_type)
#             print(available_models)
#             # Output: [('Model A', 'Version 1', '12345'), ('Model B', 'Version 2', '67890')]
#             ```

#         Notes:
#             - The `type` argument must match one of the valid `InferenceType` values in Picsellia.
#             - If no models or versions match the specified type, an empty list will be returned.
#             - Each tuple in the list provides the model name, version name, and version ID for easy identification.
#         """
#         available_models = []

#         # try:
#             # Fetch all public models
#         public_models = client.list_public_models()

#         for model in public_models:
#             # Initialize the list of versions for the model
            

#             # Fetch all model versions
#             for version in model.list_versions():
#                 if version.type.value == type:
#                     available_models.append(
#                         model.name, version.version, version.id
#                     )

#         return available_models

#         # except Exception as e:
#         #     raise ValueError(f"Failed to list object detection models: {str(e)}")


class RetrievePublicObjectDetectionModelVersionTool(Tool):
    """
    A tool to retrieve a specific Public ModelVersion from Picsellia using its name and version.

    Inputs:
        client (Client):
            - Type: `object`
            - Description: The Picsellia SDK client instance used to interact with the API.
        model_version_id (str):
            - Type: `string`
            - Description: The unique ID of the ModelVersion to retrieve.

    Output:
        output_type (ModelVersion): 
            The retrieved `ModelVersion` object.

    Example:
        ```
        # Example usage
        client = Client(api_token="your_api_token", organization_name="your_organization_name")
        model_version_id = "12345"

        # Retrieve the ModelVersion by ID
        model_version = tool.forward(client=client)
        print(model_version.name)
        ```

    Notes:
        - The tool uses the `get_model_version` method from the Picsellia SDK.
        - Proper error handling ensures that issues with retrieval are communicated effectively.
    """
    name = "retrieve_object_detection_model_version"
    description = """
    This tool retrieves a specific ModelVersion from Picsellia using its name and version. It is useful for
    accessing detailed information about a specific model version for further processing or analysis.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "The Picsellia SDK client instance used to interact with the API.",
        },
        # "model_name": {
        #     "type": "string",
        #     "description": "The public Model name",
        # },
        # "model_version": {
        #     "type": "string",
        #     "description": "The ModelVersion name"
        # }
    }
    output_type = "object"

    def forward(self, client: Client) -> ModelVersion:
        """
        Retrieves a ModelVersion from Picsellia by its unique ID.

        Args:
            client (Client): 
                An authenticated Picsellia client instance.
            model_name (str): 
                The name of the public Model to retrieve.
            model_version (str):
                The version of the public ModelVersion to retrieve.

        Returns:
            ModelVersion: The retrieved ModelVersion object.

        Raises:
            ValueError: If the ModelVersion cannot be retrieved due to an error.
        """
        # try:
            # Retrieve the model version by ID
        model = client.get_public_model(name="YoloX")
        model_version = model.get_version("YoloX_m")
        return model_version
        # except Exception as e:
        #     raise ValueError(f"Failed to retrieve ModelVersion with ID {model_name} {str(e)}")

retrieve_object_detection_model_version = RetrievePublicObjectDetectionModelVersionTool()
# list_available_public_models_for_task = PublicModelLister()
from smolagents import Tool
import os
from picsellia import Client, DatasetVersion
from picsellia.types.schemas import InferenceType

class PicselliaConnectionTool(Tool):
    name = "picsellia_client_connection"
    description = """
    This tool initializes a connection to Picsellia using an API token stored in environment variables.
    It returns a Picsellia Client instance that can be used for further operations.
    """
    inputs = {}
    output_type = "object"

    def forward(self,):
        api_token = os.getenv("PICSELLIA_TOKEN")
        if not api_token:
            raise ValueError(f"Environment variable PICSELLIA_TOKEN not found. Please set your Picsellia API token.")
        
        try:
            client = Client(api_token=api_token)
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Picsellia client: {str(e)}")

from smolagents import Tool
from picsellia import Client, DatasetVersion

class SetInferenceTypeTool(Tool):
    name = "set_inference_type"
    description = f"""
    This tool checks if a DatasetVersion has an inference type set.
    If not set, it will set it to the specified inference type.
    Returns the updated DatasetVersion.

    InferenceType is an enum of theses values {InferenceType.values()}
    """
    inputs = {
        "dataset_version": {
            "type": "object",
            "description": "The DatasetVersion object to check and update",
        },
        "inference_type": {
            "type": "object",
            "description": "The inference type to set if none exists",
        }
    }
    output_type = "object"

    def forward(self, dataset_version: DatasetVersion, inference_type: InferenceType) -> DatasetVersion:
        # try:
        #     # Get current inference type
        current_type = dataset_version.type
        
        # If no inference type is set or it's different from the desired one
        if current_type == InferenceType.NOT_CONFIGURED:
            # Set the inference type
            dataset_version.set_type(inference_type)
            print(f"Inference type set to: {inference_type}")
        else:
            print(f"Inference type already set to: {current_type}")
        
        return dataset_version
            
        # except Exception as e:
        #     raise ValueError(f"Failed to set inference type: {str(e)}")

set_inference_type_tool = SetInferenceTypeTool()
picsellia_connection_tool = PicselliaConnectionTool()


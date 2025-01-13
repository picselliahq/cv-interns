from smolagents import Tool
from picsellia.sdk.datalake import Datalake, MultiData
from typing import List, Union
import picsellia

class GetDatalakeTool(Tool):
    name = "get_datalake"
    description = """
    This tool retrieves a Datalake instance for interacting with the Picsellia Datalake.
    """
    inputs = {
        "client": {
            "type": "object",
            "description": "A Picsellia Client object"
        },
        "workspace": {
            "type": "string",
            "description": "The workspace identifier to access the correct Datalake.",
            "nullable": "True"
        },
    }
    output_type = "object"

    def forward(self, client: picsellia.Client, workspace: str = "default") -> Datalake:
        """
        Retrieves a Datalake instance using the provided API key and workspace identifier.

        Parameters:
        client (object): The Picsellia SDK client.
        workspace (str): The workspace identifier to specify the Datalake.

        Returns:
        Datalake: An instance of the Picsellia Datalake.

        Raises:
        ValueError: If authentication or workspace retrieval fails.
        """
        # try:
            # Instantiate the Datalake with the given credentials
        datalake = client.get_datalake(name=workspace)
        return datalake
        # except Exception as e:
        #     raise ValueError(f"Failed to retrieve Datalake with workspace {workspace}: {str(e)}")

intialize_datalake_tool = GetDatalakeTool()
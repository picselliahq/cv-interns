import numpy as np
from PIL import Image
import requests
import torch
from sklearn.cluster import DBSCAN
from picsellia import Client, Asset
from picsellia.exceptions import ResourceNotFoundError
from smolagents import Tool
import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import List 
import logging

class DatasetVersionEmbeddingTool(Tool):
    name = "dataset_version_outliers_detector"
    description = "A tool to compute embeddings for a dataset version and find outliers."
    inputs = {
        "client": {
            "type": "object",
            "description": "Authenticated Picsellia client instance",
        },
        "dataset_version_id": {
            "type": "string",
            "description": "ID of the dataset version to analyze",
        },
        "search_type": {
            "type": "string",
            "description": "The type of outlier detection we want to use (`centroid` or `dbscan`)",
            "nullable": "true"
        }
    }
    output_type = "object"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = "openai/clip-vit-large-patch14"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def compute_asset_embeddings(self, asset: Asset) -> np.array:
        try:
            image_url = asset.url  # Assuming each asset has a URL attribute
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Compute the embedding using the CLIP model
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error processing asset {asset.filename}: {e}")
            return None

    def forward(self, client: Client, dataset_version_id: str, search_type: str = "centroid") -> List[Asset]:
        """
        Processes a dataset version to find outlier assets using the specified search type.

        This method computes embeddings for each asset in the dataset version and then applies
        outlier detection algorithms based on the search type provided. It supports 'centroid'
        for centroid-based outlier detection and 'dbscan' for density-based spatial clustering.

        Args:
            client (Client): Authenticated Picsellia client instance.
            dataset_version_id (str): ID of the dataset version to analyze.
            search_type (str, optional): The type of outlier detection to use. Defaults to 'centroid'.

        Returns:
            List[Asset]: A list of outlier assets identified by the chosen outlier detection method.

        Raises:
            ValueError: If the dataset version with the given ID is not found.
            ValueError: If an incorrect search type is provided.
        """
        self.embeddings = []
        self.asset_ids = []

        try:
            dataset_version = client.get_dataset_version_by_id(dataset_version_id)
            assets = dataset_version.list_assets()
            for asset in assets:
                embedding = self.compute_asset_embeddings(asset)
                self.embeddings.append(embedding)
        except ResourceNotFoundError:
            raise ValueError(f"Dataset version with id {dataset_version_id} not found.")

        embeddings_array = np.array(self.embeddings)

        if search_type == 'centroid':            
            centroid = np.mean(embeddings_array, axis=0)
            centroid_distances = np.linalg.norm(embeddings_array - centroid, axis=1)
            
            outlier_threshold = np.percentile(centroid_distances, 85)
            outliers = np.where(centroid_distances > outlier_threshold)[0]
            
            outlier_assets = [assets[i] for i in outliers]
            return outlier_assets
        
        elif search_type == 'dbscan':
            dbscan = DBSCAN(metric='euclidean')
            labels = dbscan.fit_predict(embeddings_array)

            # Identify outliers (label = -1)
            outlier_indices = np.where(labels == -1)[0]
            logging.info(f"Identified {len(outlier_indices)} outliers using DBSCAN.")

            outlier_assets = [assets[i] for i in outlier_indices]
            logging.info("Outlier detection complete.")
            return outlier_assets
        else:
            raise ValueError("Incorrect search type. Options are `centroid` or `dbscan`")


class AssetsTaggingTool(Tool):
    name = "asset_tagger"
    description = "A tool to tag assets in a dataset version with one or multiple tags. It allows for the organization and categorization of assets within a dataset version for easier management and identification. This tool is particularly useful for marking assets with specific characteristics or for further processing steps."
    inputs = {
        "tags": {
            "type": "any",
            "description": "List of tags to apply to the assets",
        },
        "assets": {
            "type": "any",
            "description": "List of Asset to be tagged",
        }
    }
    output_type = "object"

    def forward(self, tags: List[str], assets: List[Asset]) -> List[Asset]:
        """
        Tags the specified assets in a dataset version with the provided tags.

        Args:
            client (Client): Authenticated Picsellia client instance.
            tags (List[str]): List of tags to apply to the assets.
            assets (List[Asset]): List of Asset to be tagged.

        Returns:
            List[Asset]: A list of assets that have been tagged.

        Raises:
            ValueError: If the dataset version with the given ID is not found.
        """
        try:
            for asset in assets:
                asset.add_tags(tags)
            return assets
        except Exception as e:
            raise ValueError(f"Could not attach tags: {e}")

asset_tagger = AssetsTaggingTool()
dataset_version_outliers_detector = DatasetVersionEmbeddingTool()
import picsellia
from transformers import CLIPProcessor, CLIPModel
from smolagents import Tool
import os
import torch 
from PIL import Image 
import numpy as np
import polars as pd
from typing import List, dict, Tuple
import requests 
import tqdm 
from picsellia.exceptions import ResourceNotFoundError
from umap import UMAP 
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

class DatasetVersionAnalyser(Tool):
    def __init__(self, dataset_version_id: str, **kwargs) -> None:
        """
        Initializes the DatasetVersionAnalyser with a specific dataset version ID.

        Args:
            dataset_version_id (str): The unique identifier for the dataset version.
            **kwargs: Additional keyword arguments passed to the parent class initializer.
        """
        super().__init__(**kwargs)
        self.client = picsellia.Client(api_token=os.environ["PICSELLIA_TOKEN"])
        self.dataset_version = self.client.get_dataset_version_by_id(dataset_version_id)
        model_name = "openai/clip-vit-large-patch14"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract_clip_embeddings(self, image: Image) -> np.array:
        """
        Extracts CLIP embeddings from a given image.

        Args:
            image (Image): The image from which to extract the embeddings.

        Returns:
            list(float): The extracted CLIP embeddings as a flattened list.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Compute the embedding using the CLIP model
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()
    
    def compute_shapes_embedding(self, asset: picsellia.Asset) -> List[dict]:
        """
        Computes the CLIP embeddings for the bounding box regions of each annotated rectangle in an asset.

        This method extracts the image regions defined by the bounding boxes of each rectangle annotation
        within an asset, computes their embeddings using the CLIP model, and returns a list of dictionaries
        containing the details of these objects.

        Args:
            asset (picsellia.Asset): The asset from which to extract and compute embeddings.

        Returns:
            List[dict]: A list of dictionaries, each containing:
                - 'id': The ID of the rectangle.
                - 'label': The label name of the rectangle.
                - 'embeddings': The computed CLIP embeddings of the object image.
                - 'bbox': The bounding box of the object in [x, y, w, h] format.
                - 'annotation_id': The ID of the annotation to which the rectangle belongs.
                - 'asset_id': The ID of the asset.

        If there are no rectangles in the asset's annotation, an empty list is returned.
        """
        try:
            annotation = asset.get_annotation()
            rectangles = annotation.list_rectangles()
        except ResourceNotFoundError as e:
            return []
        
        if len(rectangles) < 1: return []

        orig_im = Image.open(requests.get(asset.url, stream=True).raw).convert("RGB")
        objects_details = []
        print(f"Extracting {len(rectangles)} objects CLIP embeddings.")
        for rectangle in rectangles:
            x, y, w, h = rectangle.x, rectangle.y, rectangle.w, rectangle.h
            object_image = orig_im.copy().crop((x, y, x + w, y + h))
            embeddings = self.extract_clip_embeddings(object_image)
            objects_details.append(
                {
                    'id': rectangle.id,
                    'label': rectangle.label.name,
                    'embeddings': embeddings,
                    'bbox': [x, y, w, h], # x, y, w, h formatting like picsellia boxes.
                    'annotation_id': str(annotation.id),
                    'asset_id': str(asset.id)
                }
            )
        return objects_details
    
    def compute_image_embedding(self, asset: picsellia.Asset) -> dict:
        """
        Computes the CLIP embeddings for the entire image of an asset.

        Args:
            asset (picsellia.Asset): The asset from which to extract and compute the image embeddings.

        Returns:
            dict: A dictionary containing the asset ID, filename, embeddings, asset tags, and data tags.
        """
        image = Image.open(requests.get(asset.url, stream=True).raw).convert("RGB")
        embeddings = self.extract_clip_embeddings(image)
        return {
            'asset_id': str(asset.id),
            'filename': asset.filename,
            'embeddings': embeddings,
            'asset_tags': [e.name for e in asset.get_tags()],
            'data_tags': [e.name for e in asset.get_data_tags()]
        }

    def generate_raw_analytics_data(self, with_object: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates raw analytics data for the dataset version by extracting CLIP embeddings for each image and object.

        This method iterates over all assets in the dataset version, computes the CLIP embeddings for each image,
        and optionally for each object within the images if `with_object` is set to True. The embeddings are then
        stored in two separate pandas DataFrames, one for image data and one for objects data, which are subsequently
        saved to parquet files.

        Args:
            with_object (bool): A flag to determine whether to compute embeddings for objects within the images.
                                Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas DataFrames. The first DataFrame contains
                                               image data, and the second contains objects data if `with_object` is True.
        """
        print(f"Extracting CLIP embeddings for dataset {self.dataset_version.name}/{self.dataset_version.version} ...")
        image_data_list = []
        objects_data_list = []
        for asset in tqdm.tqdm(self.dataset_version.list_assets()):
            image_data = self.compute_image_embedding(asset=asset)
            image_data_list.append(image_data)
            if with_object:
                objects_data = self.compute_shapes_embedding(asset)
                objects_data_list.extend(objects_data)
        
        # Convert lists to DataFrames
        self.image_data_df = pd.DataFrame(image_data_list)
        self.objects_data_df = pd.DataFrame(objects_data_list)

        # Save DataFrames to parquet files
        self.image_data_df.to_parquet(f"{self.dataset_version.name}_{self.dataset_version.version}_image_data.parquet")
        if with_object:
            self.objects_data_df.to_parquet(f"{self.dataset_version.name}_{self.dataset_version.version}_objects_data.parquet")

        return

    def generate_umap_view(self, type: str = "image", fpath: str = None,  n_neighbors: int = 15, 
                              min_dist: float = 0.1,
                              random_state: int = 42,
                              figsize: tuple = (12, 8),
                              analyze_clusters: bool = True) -> str:

        if type == "image":
            if isinstance(self.image_data_df, pd.DataFrame):
                df = self.image_data_df
            else:
                df = pd.read_parquet(fpath)
        
        elif type == "object":
            if isinstance(self.objects_data_df, pd.DataFrame):
                df = self.objects_data_df
            else:
                df = pd.read_parquet(fpath)

        mask = [len(emb) > 0 for emb in df['embeddings']]
        df_filtered = df.filter(pd.Series(mask))
        
        print(f"Original number of {type}: {len(df)}")
        print(f"Rows after filtering empty embeddings: {len(df_filtered)}")
        
        # Convert the filtered embeddings from lists to a numpy array
        embeddings = np.stack(df_filtered.select('embeddings').to_numpy().flatten())
        
        # Perform UMAP dimensionality reduction
        print("Fitting UMAP transformation...")
        umap_reducer = UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state)
        embeddings_2d = umap_reducer.fit_transform(embeddings)
        
        # Calculate point density for coloring
        if analyze_clusters:
            print("Analyzing local density patterns...")
            nbrs = NearestNeighbors(n_neighbors=min(50, len(embeddings)))
            nbrs.fit(embeddings_2d)
            distances, _ = nbrs.kneighbors(embeddings_2d)
            density_scores = np.mean(distances, axis=1)
            density_scores = 1 / (1 + density_scores)  # Transform to make dense regions = high values
        else:
            density_scores = None
        
        # Create the visualization
        plt.figure(figsize=figsize)
        
        if analyze_clusters:
            # Create a scatter plot colored by density
            scatter = plt.scatter(embeddings_2d[:, 0], 
                                embeddings_2d[:, 1],
                                alpha=0.6,
                                s=30,
                                c=density_scores,
                                cmap='viridis',
                                rasterized=True)
            plt.colorbar(scatter, label='Local Density')
        else:
            # Create a simple scatter plot
            plt.scatter(embeddings_2d[:, 0], 
                    embeddings_2d[:, 1],
                    alpha=0.5,
                    s=20,
                    color='#1f77b4',
                    rasterized=True)
        
        # Enhance plot appearance
        plt.title("UMAP Visualization of {type} Embeddings", fontsize=14, pad=20)
        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        
        # Clean up the plot
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        figpath = f"{type}_embeddings_umap_representation.png"
        plt.savefig(figpath)
        
        return figpath
    

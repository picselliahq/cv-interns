import os
import torch
from open_clip import tokenize, load_model
from sklearn.cluster import KMeans
from picsellia import Client
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm


def compute_embeddings_and_extract_outliers(dataset_version_id, api_token, device='cpu'):
    # Initialize Picsellia client
    client = Client(api_token=api_token)
    dataset_version = client.DatasetVersion(id=dataset_version_id)

    # Download dataset assets
    dataset_version.download()

    # Load the CLIP model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, preprocess = load_model('ViT-B/32', device=device)

    # Prepare images and compute embeddings
    embeddings = []
    asset_ids = []
    for asset in tqdm(dataset_version.list_assets()):
        image_path = os.path.join(dataset_version.name, asset.filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        embeddings.append(embedding.cpu().numpy())
        asset_ids.append(asset.id)

    # Normalize embeddings
    embeddings = torch.tensor(embeddings).squeeze(1)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)

    # Perform KMeans clustering
    num_clusters = min(len(embeddings) // 2, 10)  # Limit the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

    # Find outliers based on distance from cluster centroids
    distances = kmeans.transform(embeddings)
    max_distance = distances.max(axis=1)
    threshold = np.percentile(max_distance, 95)  # Set threshold as 95th percentile of max distances
    outliers = np.where(max_distance > threshold)[0]

    # Tag assets as outliers
    for i in outliers:
        asset = client.Asset(id=asset_ids[i])
        asset.add_tag("outlier")

    print(f"Tagged {len(outliers)} assets as outliers.")



def compute_clip_embeddings_and_tag_outliers(dataset_version_id, api_token, clip_model_path, device='mps'):
    # Initialize Picsellia client
    client = Client(api_token=api_token)
    dataset_version = client.DatasetVersion(id=dataset_version_id)

    # Download dataset assets
    dataset_version.download()

    # Get the list of labels and determine the number of clusters
    labels = dataset_version.list_labels()
    num_clusters = len(labels)

    # Load the CLIP model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, preprocess = load_model(clip_model_path, device=device)

    # Prepare images and compute embeddings
    embeddings = []
    asset_ids = []
    for asset in tqdm(dataset_version.list_assets()):
        image_path = os.path.join(dataset_version.name, asset.filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        embeddings.append(embedding.cpu().numpy())
        asset_ids.append(asset.id)

    # Normalize embeddings
    embeddings = torch.tensor(embeddings).squeeze(1)
    embeddings /= embeddings.norm(dim=-1, keepdim=True)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

    # Find outliers based on cluster sizes
    cluster_sizes = torch.bincount(torch.tensor(kmeans.labels_))
    small_clusters = torch.where(cluster_sizes < cluster_sizes.mean())[0]

    # Tag assets as outliers if they belong to small clusters
    for i, label in enumerate(kmeans.labels_):
        if label in small_clusters:
            asset = client.Asset(id=asset_ids[i])
            asset.add_tag("outlier")

    print(f"Tagged {len(small_clusters)} assets as outliers.")

# Example usage:
# compute_clip_embeddings_and_tag_outliers(dataset_version_id="your_dataset_version_id", api_token="your_api_token", clip_model_path="path_to_your_clip_model")




from typing import List

def create_dataset_prompt(name: str, tag: str, labels: List[str]) -> str:
    prompt = f"""
    "Using Picsellia, create a Dataset object named {name} containing all data items tagged with {tag}. 
    Once the dataset is created, use the zero_shot_object_detector tool to pre-annotate the data by adding 
    bounding boxes around objects detected as {" ,".join(label for label in labels)}. 
    Focus on executing the task using the appropriate Picsellia objects and methods without providing explanations or demonstrations."
    """
    return prompt
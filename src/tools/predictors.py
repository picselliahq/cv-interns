import picsellia 
import picsellia.types
import picsellia.types.enums
import picsellia.types.schemas
from picsellia import Label, Asset
import requests
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Generator
import time 
from smolagents import tools, Tool
import numpy as np
from utils.label import find_picsellia_label



class ZeroShotDetectorTool(Tool):
    name = "zero_shot_object_detector_tool"
    description = """
    This is a tool that runs zero-shot object detection on a single asset using provided labels.
    It performs Owl-ViT-based object detection and stores the predictions (bounding boxes, 
    confidence scores) as annotations in Picsellia, complete with bounding-box coordinates 
    and the duration of the inference step.
    """
    inputs = {
        "labels": {
            "type": "object",
            "description": "List of Picsellia Label objects to detect",
        },
        "asset": {
            "type": "object",
            "description": "A Picsellia asset object on which to run zero-shot detection",
        }
    }
    output_type = "string"

    def forward(self, labels: List[Label], asset: Asset) -> str:
        start = time.time()

        # Initialize model and processor
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        model.to(device='mps')

        # Prepare image and text inputs
        image = Image.open(requests.get(asset.url, stream=True).raw)
        texts = [[f"a photo of a {label.name}" for label in labels]]

        # Process inputs
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs.to(device="mps")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )

        # Create annotation if detections exist
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        detected_labels = results[0]["labels"]
        elapsed = time.time() - start
        image_width, image_height = image.size

        if len(boxes) > 0:
            # Calculate average box area
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            avg_area = sum(areas) / len(areas)

            # Define size thresholds based on average area
            MIN_BOX_AREA = 0.5 * avg_area
            MAX_BOX_AREA = 2.0 * avg_area

            annotation = asset.create_annotation(duration=elapsed)
            rectangles = []
            # Process each detection
            for box, score, label_idx in zip(boxes, scores, detected_labels):
                if round(score.item(), 3) > 0.15:
                    box = [int(coord) for coord in box.tolist()]
                    x = np.clip(box[0], 0, image_width - 1)  # Ensure x is within the image width
                    y = np.clip(box[1], 0, image_height - 1)  # Ensure y is within the image height
                    w = np.clip(box[2], 0, image_width - 1) - x  # Clip the width and adjust relative to x
                    h = np.clip(box[3], 0, image_height - 1) - y

                    # Calculate area of the current box
                    area = w * h

                    # Filter out boxes that are outside the area thresholds
                    if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
                        continue

                    # Get corresponding Picsellia label
                    text = texts[0][label_idx]
                    pic_label = find_picsellia_label(labels, text)
                    # Create rectangle annotation
                    rectangles.append((int(x), int(y), int(w), int(h), pic_label))

            if rectangles:
                annotation.create_multiple_rectangles(rectangles=rectangles)

        return f"Detection completed. Processing time: {elapsed:.2f} seconds"

   



zero_shot_object_detector = ZeroShotDetectorTool()


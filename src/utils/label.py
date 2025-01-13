import picsellia
from typing import List

def find_picsellia_label(labels: List[picsellia.Label], vlm_answer: str) -> picsellia.Label:
        text_label = vlm_answer.split(" ")[-1]
        for label in labels:
            if text_label == label.name:
                return label 
        return None
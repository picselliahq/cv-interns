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

from smolagents import CodeAgent, HfApiModel, LiteLLMModel, DuckDuckGoSearchTool, ToolCallingAgent, GradioUI
from agents.interns.data_scientist import managed_data_scientist_intern
from agents.interns.data_scientist import toolset as data_scientist_toolset

from agents.interns.data_engineer import managed_data_engineer_intern
from agents.interns.data_engineer import toolset as data_engineer_toolset
# model = LiteLLMModel(model_id="openai/gpt-4o", api_key="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")


model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct", token="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")

picsellia_ai_hr_workforce = CodeAgent(
    tools=data_engineer_toolset+data_scientist_toolset,
    model=model, 
    # managed_agents=[managed_data_scientist_intern, managed_data_scientist_intern]
)
picsellia_ai_hr_workforce.run(
    "pre-annotate cars, people, and bikes in the dataset version  `thibaut-le-boss/initial` "
)





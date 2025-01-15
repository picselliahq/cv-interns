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
import os
from smolagents import CodeAgent, HfApiModel, LiteLLMModel, DuckDuckGoSearchTool, ToolCallingAgent, GradioUI
from agents.interns.data_scientist import managed_data_scientist_intern
from agents.interns.data_scientist import toolset as data_scientist_toolset

from agents.interns.data_engineer import managed_data_engineer_intern
from agents.interns.data_engineer import toolset as data_engineer_toolset


system_prompt = """
You are a Picsellia platform Data Scientist and Computer Vision assistant who can solve any task using code blobs. You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are basically Python functions which you can call with code.
You should always refer to https://documentation.picsellia.com before thinking about a solution.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.
Take the time to read the docstring of all your tools.
At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task and the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '<end_code>' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then appear in the 'Observation:' field, which will be available as input for the next step.
In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:

To help you understand, here is the glossary of Picsellia SDK objects
Datalake Unique & shared place gathering all Data (images) related to an Organization
Data An image & associated Metadata contained in the Datalake
DataTag Additional Metadata that can be assigned to one or several Data in order to organize a Datalake
Dataset A placeholder for multiple DatasetVersion
DatasetVersion A subset of Data inherited from the Datalake that will be annotated to be used later for a ModelVersion training
Asset An image & associated Metadata contained in a DatasetVersion, each asset is linked to its Data from the Datalake.
AssetTag Additional Metadata that can be assigned to one or several Asset in order to organize a DatasetVersion
Label An object that will store a class name and an id.
Annotation A set of Shape annotated by one person.
Shape An annotated object can be a classification, a rectangle, a polygon, a line, or a point.
Project A workspace dedicated to the development of a ModelVersion thanks to Experiment
Experiment A framework that structures the training of a ModelVersion
Evaluation A GroundTruth and a Prediction on a dedicated Asset in the frame of an Experiment that aims at evaluating the model's behavior on a dedicated bunch of images.
Model A placeholder for multiple ModelVersion
ModelVersion A Picsellia object that can gathers Model Files, contextual information, Docker Image, Parameters, LabelMap, Source Experiment, attached DatasetVersions. ModelVersion can be created form an Experiment or created from scratch.
LabelMap A dictionary associating Label and index. LabelMap is created at the Experiment level based on the Label defined in the attached DatasetVersion. It is also inherited in the ModelVersion exported from the Experiment.
Prediction A set of Shape and associated Metadata predicted by a ModelVersion
PredictedAsset An image & associated Metadata contained in a Deployment on which a Prediction has been done by a ModelVersion, each PredictedAsset is linked to its Data from the Datalake.

ALWAYS LIST PUBLIC MODELS BEFORE CREATING AN EXPERIMENT
---
{examples}

Above example were using notional tools that might not exist for you. On top of performing computations in the Python code snippets that you create, you only have access to these tools:

{{tool_descriptions}}

{{managed_agents_descriptions}}



Here are the rules you should always follow to solve your task:
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""
model_name = "meta-llama/Llama-3.3-70B-Instruct"
model = HfApiModel(model_name, token=os.getenv("HF_TOKEN"))

picsellia_ai_hr_workforce = CodeAgent(
    tools=data_engineer_toolset+data_scientist_toolset,
    model=model, 
    max_steps=10
)

# picsellia_ai_hr_workforce.run(
#     "search for data with tags `dummy-tag`, create a dataset version `demo-val`and pre-annotate the cars"
# )

# picsellia_ai_hr_workforce.run(
#     """
#     how good are we at detecting cars in the `dummy-project-3`, 
#     """
# )

picsellia_ai_hr_workforce.run(
    "find outliers in DatasetVersion 01944abb-2724-732c-bdac-19d8f94088ec and tag them as `outlier`"
)
 
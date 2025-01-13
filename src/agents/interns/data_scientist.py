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
from smolagents import tools, Tool, ManagedAgent
import numpy as np

from smolagents import CodeAgent, HfApiModel, LiteLLMModel, DuckDuckGoSearchTool
from tools.dataset.read import list_dataset_assets_tool, list_dataset_labels_tool, fetch_dataset_version_by_id_tool,check_if_label_exists
from tools.dataset.init import set_inference_type_tool, picsellia_connection_tool
from tools.dataset.create import create_train_test_val_dataset_version
from tools.project.write import create_picsellia_project, attach_dataset_to_project
from tools.experiment.write import create_picsellia_experiment
from tools.models.search import  retrieve_object_detection_model_version # list_available_public_models_for_task
from tools.experiment.actions import launch_training_tool
from tools.experiment.write import create_picsellia_experiment, attach_model_version_to_experiment, attach_dataset_to_experiment

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



dataset_toolset = [
    picsellia_connection_tool,
    list_dataset_assets_tool, list_dataset_labels_tool, fetch_dataset_version_by_id_tool,
    check_if_label_exists, 
    set_inference_type_tool, picsellia_connection_tool,
    create_train_test_val_dataset_version,
]

project_tool_set = [
    create_picsellia_project, attach_dataset_to_project,
    retrieve_object_detection_model_version, launch_training_tool,
    create_picsellia_experiment,attach_dataset_to_experiment,
    attach_model_version_to_experiment, create_train_test_val_dataset_version
]

toolset = dataset_toolset + project_tool_set

model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct", token="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")

data_scientist_intern = CodeAgent(
    model=model,
    tools=toolset,
    system_prompt=system_prompt,
    max_steps=10,
)

managed_data_scientist_intern = ManagedAgent(    
    name="Model training related taks",
    agent=data_scientist_intern, 
    description="Performs any Data Science and model creation related tasks on Picsellia"
)
# cv-interns
A repo centralizing the AI agents able to build computer vision models with Picsellia

# Picsellia AI Workforce

This project implements an AI-powered workforce using the Picsellia platform for automated dataset annotation. It utilizes various AI models and tools to pre-annotate objects like cars, people, and bikes in datasets.

## Features

- Automated dataset annotation using AI models
- Integration with Picsellia platform
- Support for multiple object detection tasks
- Modular architecture with specialized AI "interns" (Data Scientist and Data Engineer)

## Prerequisites

- Python 3.8+
- Picsellia account and API access
- Hugging Face API token
- Required Python packages (see Installation section)

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Required Packages

- picsellia
- torch
- transformers
- Pillow
- requests
- smolagents
- numpy

## Configuration

Before running the application, make sure to:

1. Set up your Hugging Face API token
2. Configure your Picsellia credentials
3. Ensure access to the required datasets

## Usage

Basic usage example:

```python
from agent import picsellia_ai_hr_workforce

# Run annotation task
picsellia_ai_hr_workforce.run(
    "pre-annotate cars, people, and bikes in the dataset version `your-dataset-name`"
)
```

## Project Structure

- `src/agent.py`: Main agent implementation
- `agents/interns/`: Specialized AI agents
  - `data_scientist/`: Data Scientist intern implementation
  - `data_engineer/`: Data Engineer intern implementation

## Models

The project uses:
- Llama 3.3 70B Instruct model for main processing
- Owlv2 for object detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache-2

## Support

For support, please contact thibaut@picsellia.com



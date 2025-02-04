from smolagents import Tool, LiteLLMModel, CodeAgent
import os 

model = LiteLLMModel(model_id="openai/gpt-4o", api_key=os.environ["OPENAI_API_KEY"])

system_prompt = """
Analyze the sent dict file to generate a comprehensive and structured report. The report should provide detailed insights into the data science activities, explain key concepts, and offer visualizations to enhance understanding. Use the following structure and guidelines for the output:

1. Project Overview
    Summarize the project details, including:
        Project name and description.
        Creation and update dates.
        List of collaborators and their roles.
        Number of experiments and datasets associated with the project.
        Provide a clear explanation of the project's significance or goals.
2. Dataset Analysis

    For each dataset:
        Name the dataset and summarize its key properties (e.g., labels, number of images, objects, and annotations).
        Provide a breakdown of label distribution (e.g., objects per class).
        Identify any imbalances or issues in the dataset that might affect model performance.
        Include visualizations such as:
        Pie charts or bar charts for label distributions.
        Summary tables with dataset statistics.

3. Experiment Analysis
    Detail each experiments:
        Experiment name, description, and associated datasets.
        Model details (e.g., architecture, version, and base model used).
        Key parameters (e.g., epochs, batch size, learning rate).
        Explain the significance of each parameter and its potential impact on training.
        Include training metrics:
        Precision, recall, AP (Average Precision), and AR (Average Recall).
        True positives, false positives, and false negatives per class.
        Visualize important trends:
        Line graphs for loss metrics over epochs.
        Precision-recall curves for individual classes.

4. Performance Insights
    Summarize model performance:
    Compare metrics across classes (e.g., AP, AR).
    Highlight areas of strength and areas for improvement.
    Explain key terms (e.g., AP, IoU, mAP) in a glossary section to ensure clarity.
    Identify potential reasons for performance discrepancies (e.g., data imbalance, model settings).

5. Recommendations
    Provide actionable suggestions for improvement, such as:
    Balancing datasets or adding more training data for underrepresented classes.
    Adjusting model parameters (e.g., learning rate, batch size).
    Experimenting with different architectures or augmentations.
    Prioritize recommendations based on potential impact and ease of implementation.
    Give precise intructions as if you were talking to an intern. 


"""


import argparse
import os
from typing import List, Iterable
import json
from langchain_community.document_loaders import GitLoader, SitemapLoader
from langchain.schema import Document
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.retrievers import BM25Retriever

from smolagents import Tool, CodeAgent, HfApiModel, ToolCallingAgent
import os
os.environ['USER_AGENT'] = 'myagent'

def create_doc_from_html(web_path: str, filter_urls: List[str]) -> List[Document]:
    print("ici")
    loader = SitemapLoader(
        web_path=web_path,
        filter_urls=filter_urls,
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


def create_doc_from_git(clone_url: str, repo_path: str, branch: str) -> List[Document]:
    print("la")
    loader = GitLoader(
        clone_url=clone_url,
        repo_path=repo_path,
        branch=branch,
        file_filter=lambda file_path: file_path.endswith(".py"),
    )
    docs = loader.load()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    return python_splitter.split_documents(docs)


def extract_knowledge(db_type):
    documents = []

    # if db_type == "sdk":
    #     print("here")
    documents += create_doc_from_html(
        web_path="https://documentation.picsellia.com/sitemap.xml",
        filter_urls=[
            "^https://documentation\.picsellia\.com/reference/(client|datalake|data|multidata|datasource|tag|dataset|datasetversion|asset|multiasset|worker|annotation|rectangle|polygon|point|line|classification|label|model|modelversion|modelfile|modelcontext|project|experiment|artifact|log|deployment|job|changelog|index)$",
        ],
    )

    # elif db_type == "doc":
    documents += create_doc_from_html(
        web_path="https://documentation.picsellia.com/sitemap.xml",
        filter_urls=["https://documentation.picsellia.com/docs/.*"],
    )

    # elif db_type == "training_engine":
    # documents += create_doc_from_git(
    #     clone_url="https://github.com/picselliahq/picsellia-training-engine",
    #     repo_path="picsellia-training-engine",
    #     branch="master",
    # )

    return documents


    # db = Chroma.from_documents(
    #     documents, embedding=embedding, persist_directory="{}_db".format(db_type)
    # )
def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(load_docs_from_jsonl('data.jsonl'))

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of picsellia documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

retriever_tool = RetrieverTool(docs_processed)
# if __name__ == "__main__.py":

# document = extract_knowledge("sdk")
# save_docs_to_jsonl(document,'data.jsonl')


    # docs2=load_docs_from_jsonl(,'data.jsonl')

system_prompt = """
You are a Knowledge expert at Picsellia, 

Your Job is to provide a step by step procedure to follow to achieve a task with Picsellia SDK.

you goal is to create a roadmap for another AI agent that will be coding the solution.

"""
agent = ToolCallingAgent(
    tools=[retriever_tool], 
    model=HfApiModel("meta-llama/Llama-3.3-70B-Instruct",
    token="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"),
    max_steps=4,
)

agent.run(
    "what are the necessary steps to train a model with picsellia?"
)
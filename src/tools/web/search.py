import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool, Tool
from typing import List
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.retrievers import BM25Retriever
# from langchain.document_loaders import SitemapLoader
# from bs4 import BeautifulSoup
# from langchain.document_loaders import GitLoader, SitemapLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.schema import Document
# from langchain.text_splitter import (
#     Language,
#     RecursiveCharacterTextSplitter,
# )
# from langchain.vectorstores.chroma import Chroma

# def create_doc_from_html(web_path: str, filter_urls: List[str]) -> List[Document]:
#     loader = SitemapLoader(
#         web_path=web_path,
#         filter_urls=filter_urls,
#         # parsing_function=remove_unwanted_html,
#     )
#     docs = loader.load()

#     python_splitter = RecursiveCharacterTextSplitter.from_language(
#         language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
#     )
#     return python_splitter.split_documents(docs)

# source_docs = [
#     Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
#     for doc in knowledge_base
# ]

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50,
#     add_start_index=True,
#     strip_whitespace=True,
#     separators=["\n\n", "\n", ".", " ", ""],
# )
# docs_processed = text_splitter.split_documents(source_docs)

# from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
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
class PicselliaDocumentationBrowser(Tool):
    name = "browse_picsellia_documentation"
    description = """
    browse Picsellia sdk reference documentation
    """
    inputs = {
        
    }
    output_type = "string"

    def forward(self,) -> str:
        """Visits a webpage at the given URL and returns its content as a markdown string.

        Args:
            url: The URL of the webpage to visit.

        Returns:
            The content of the webpage converted to Markdown, or an error message if the request fails.
        """
        try:
            # Send a GET request to the URL
            response = requests.get("https://documentation.picsellia.com/reference/client")
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return markdown_content

        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
    
browse_picsellia_documentation = PicselliaDocumentationBrowser()
"""
module of common tools to use
"""


import chromadb
from .utils import tool_call


def make_chroma_search_tool(path: str, collection_name: str):
    """
    Creates a tool to use ChromaDB to do a similarity search
    Parameters
    ----------
    path : str
        path to vector db
    collection_name : str
        the name of the collection
    Returns
    -------
    chroma_search_tool:
        callable that can be used in a FunctionalNode
    """
    # get a client up and connected to the vector db and the collection
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)

    # create a tool call with retrieval_args and retrieved_results as the output
    @tool_call(input_key="retrieval_args", output_key="retrieved_results")
    def chroma_search_tool(query: str, n_results: int = 3) -> dict[str, dict]:
        # query the collection and return the results
        results = collection.query(query_texts=[query], n_results=n_results)
        return results

    return chroma_search_tool

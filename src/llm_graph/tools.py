import chromadb
from .utils import tool_call

def make_chroma_search_tool(
    path: str,
    collection_name: str
):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    @tool_call(input_key="retrieval_args", output_key="retrieved_results")
    def chroma_search_tool(query: str, n_results: int = 3):
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

    return chroma_search_tool
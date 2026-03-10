from llm_graph.core.nodes import FunctionalNode
from llm_graph.tools import make_chroma_search_tool
def create_retrieval_node(
    path: str,
    collection_name: str,
    query_key: str = "user_query",
    name: str = "retrieval",
    next_node_name: str | None = None
):
    tool = make_chroma_search_tool(path=path, collection_name=collection_name)
    def func(state: dict):
        query = state.get(query_key, "")
        ret_key = {"retrieval_args" : {"query": query}}
        resp = tool(ret_key)
        if resp['retrieved_results_success']:
                docs = resp["retrieved_results"].get("documents", [[]])[0]
                combined = "\n\n".join(docs) if docs else ""
        else:
            combined = ""
        return resp | {"inject_text": combined}
    node = FunctionalNode(func=func, name=name, next_node_name=next_node_name)
    return node
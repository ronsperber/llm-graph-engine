import chromadb
def chroma_retrieval_fn(
        path: str,
        collection_name: str,
):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    def _call(
            query: str,
            history: list[str] | None = None,
            n_results: int = 3
    ):
        if history is not None:
            history_txt = " ".join(history)
        else:
            history_txt = ""
        hist_query = history_txt + query
        results = collection.query(
            query_texts=[hist_query],
            n_results=n_results
        )
        scores = (results.get("distances") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]

        if documents is None:
            documents = []
            scores = []
        return {"documents":documents, "scores": scores}
    
    return _call
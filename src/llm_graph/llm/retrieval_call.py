from typing import Callable
class RetrieverCall:
    def __init__(
        self,
        retrieval_fn: Callable,
        n_results: int = 3,
        history: list[str] | None = None,
        query_key: str = "user_query"
    ):
        self.retrieval_fn = retrieval_fn
        self.query_key = query_key
        self.history = history
        self.n_results = n_results
    def __call__(self, state: dict):
        if self.query_key in state:
            query = state[self.query_key]
        else:
            raise KeyError (f"{self.query_key} was missing from the keys")
        results = self.retrieval_fn(
            query=query,
            n_results=self.n_results,
            history=self.history
        )
        return {"retrieved_results" : results}



    
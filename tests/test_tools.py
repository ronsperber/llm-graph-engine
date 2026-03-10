from unittest.mock import MagicMock, patch

from llm_graph.tools import make_chroma_search_tool


@patch("llm_graph.tools.chromadb.PersistentClient")
def test_make_chroma_search_tool_calls_chromadb_and_returns_results(mock_client_cls):
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"docs": ["doc1"]}

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client_cls.return_value = mock_client

    tool = make_chroma_search_tool(path="/tmp/chroma", collection_name="test_collection")

    state = {"retrieval_args": {"query": "hello", "n_results": 2}}
    result = tool(state)

    mock_client_cls.assert_called_once_with(path="/tmp/chroma")
    mock_client.get_collection.assert_called_once_with("test_collection")
    mock_collection.query.assert_called_once_with(query_texts=["hello"], n_results=2)

    assert result["retrieved_results"] == {"docs": ["doc1"]}
    assert result["retrieved_results_success"] is True


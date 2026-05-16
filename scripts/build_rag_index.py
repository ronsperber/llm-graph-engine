from pathlib import Path
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DOC_PATH = Path("docs")
DB_PATH = "vector_db"

# Embedding model (small + fast demo model)
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Recreate persistent client
client = chromadb.PersistentClient(path=DB_PATH)

# Delete and recreate collection for clean rebuild
try:
    client.delete_collection("llm_graph_docs")
except Exception:
    pass

collection = client.create_collection(
    name="llm_graph_docs", embedding_function=embedding_fn # type: ignore (issue with chromadb type stubs)
)


def split_markdown_sections(text: str):
    """
    Split corpus by markdown headers (## or ### etc).
    """
    sections = re.split(r"\n(?=#{2,})", text)
    return [s.strip() for s in sections if s.strip()]


def normalize_chunk(section_text: str, component_name: str | None):
    """
    Add semantic anchor line if component is known.
    """

    if component_name:
        prefix = f"Component documentation: {component_name}\n\n"
        return prefix + section_text

    return section_text


def build_index():
    doc_id = 0

    for file in DOC_PATH.rglob("*.md"):
        text = file.read_text(encoding="utf-8")

        sections = split_markdown_sections(text)
        print(f"{file} has {len(sections)} sections")
        current_component = None

        for section in sections:
            match = re.search(r"Component:\s*(.+)", section)
            if match:
                current_component = match.group(1).strip()

            normalized_chunk = normalize_chunk(section, current_component)

            metadata = {
                "source": str(file),
                "component": current_component or "unknown",
                "section": "documentation",
            }

            chunk_id = f"{file.stem}_{doc_id}"
            print(f"Processing chunk_id : {chunk_id}")
            collection.add(
                documents=[normalized_chunk], ids=[chunk_id], metadatas=[metadata]
            )

            doc_id += 1


if __name__ == "__main__":
    build_index()

from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DOC_PATH = Path("docs")
DB_PATH = "vector_db"

print("Working directory:", Path.cwd())
print("Docs path exists:", DOC_PATH.exists())

# small, fast embedding model
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name="llm_graph_docs",
    embedding_function=embedding_fn
)


def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


doc_id = 0

for file in DOC_PATH.rglob("*.md"):
    print("Found:", file)
    text = file.read_text()

    chunks = chunk_text(text)

    for chunk in chunks:
        collection.add(
            documents=[chunk],
            ids=[f"doc_{doc_id}"],
            metadatas=[{
                "source": str(file),
                "type": file.parent.name
            }]
        )
        doc_id += 1

print(f"Ingested {doc_id} chunks.")
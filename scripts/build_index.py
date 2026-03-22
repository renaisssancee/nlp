"""Build ChromaDB vector index from processed movies.jsonl."""

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

MOVIES_FILE = PROJECT_ROOT / config["data"]["movies_file"]
CHROMA_PATH = PROJECT_ROOT / config["chroma_db_path"]
COLLECTION_NAME = config["chroma_collection"]
EMBEDDING_MODEL = config["embedding_model"]


def load_movies(path: Path) -> list[dict]:
    """Load movies from JSONL file."""
    movies = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            movies.append(json.loads(line))
    return movies


def build_index():
    print(f"Loading movies from {MOVIES_FILE}...")
    movies = load_movies(MOVIES_FILE)
    print(f"Loaded {len(movies)} movies")

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    texts = [m["text_for_embedding"] for m in movies]
    print(f"Generating embeddings for {len(texts)} documents (batch_size=64)...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Create ChromaDB collection
    print(f"Creating ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add documents in batches
    batch_size = 500
    for i in range(0, len(movies), batch_size):
        batch = movies[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size].tolist()

        ids = [m["id"] for m in batch]
        documents = [m["text_for_embedding"] for m in batch]
        metadatas = [
            {
                "title": m["title"],
                "year": int(m["year"]),
                "duration_min": int(m["duration_min"]),
                "rating": float(m["rating"]),
                "genres": m["genres_pipe"],
                "tags": m["tags_pipe"],
                "overview": m["overview"],
            }
            for m in batch
        ]

        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"  Added batch {i // batch_size + 1}: {len(batch)} documents")

    print(f"Total documents in collection: {collection.count()}")

    # Test query
    print("\nTest query: 'space exploration emotional drama'")
    results = collection.query(
        query_embeddings=model.encode(["space exploration emotional drama"]).tolist(),
        n_results=5,
    )
    for i, (doc_id, metadata) in enumerate(zip(results["ids"][0], results["metadatas"][0])):
        print(f"  {i + 1}. {metadata['title']} ({metadata['year']}) - rating: {metadata['rating']}")

    print("\nIndex built successfully!")


if __name__ == "__main__":
    build_index()

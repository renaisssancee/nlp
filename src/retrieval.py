"""Retrieval module: vector search with ChromaDB + cross-encoder reranking."""

from pathlib import Path

import chromadb
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)


class Retriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config["embedding_model"])
        self.reranker = CrossEncoder(config["reranker_model"])

        chroma_path = PROJECT_ROOT / config["chroma_db_path"]
        client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = client.get_collection(config["chroma_collection"])

        self.n_results = config["retrieval"]["n_results"]
        self.top_k = config["retrieval"]["top_k"]
        self.min_results = config["retrieval"]["min_results_with_filter"]

    def _build_where_filter(self, parsed_query: dict) -> dict | None:
        """Build ChromaDB where filter from parsed query."""
        conditions = []

        genre = parsed_query.get("genre")
        if genre:
            conditions.append({"genres": {"$contains": genre}})

        max_duration = parsed_query.get("max_duration")
        if max_duration:
            conditions.append({"duration_min": {"$lte": int(max_duration)}})

        min_year = parsed_query.get("min_year")
        if min_year:
            conditions.append({"year": {"$gte": int(min_year)}})

        min_rating = parsed_query.get("min_rating")
        if min_rating:
            conditions.append({"rating": {"$gte": float(min_rating)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(self, parsed_query: dict) -> list[dict]:
        """Retrieve and rerank movies based on parsed query."""
        semantic_query = parsed_query.get("semantic_query", "")
        query_embedding = self.embedding_model.encode([semantic_query]).tolist()

        where_filter = self._build_where_filter(parsed_query)


        results = self._query_chroma(query_embedding, where_filter)


        if where_filter and len(results) < self.min_results:
            print(f"Only {len(results)} results with filter, retrying without filter...")
            results = self._query_chroma(query_embedding, where_filter=None)

        if not results:
            return []


        reranked = self._rerank(semantic_query, results)
        return reranked[: self.top_k]

    def _query_chroma(self, query_embedding: list, where_filter: dict | None) -> list[dict]:
        """Query ChromaDB and return results."""
        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": self.n_results,
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            print(f"ChromaDB query error: {e}")

            if where_filter:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=self.n_results,
                )
            else:
                return []

        movies = []
        if not results["ids"] or not results["ids"][0]:
            return movies

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 1.0
            similarity = 1.0 - distance  
            metadata = results["metadatas"][0][i]
            movies.append({
                "id": doc_id,
                "title": metadata["title"],
                "year": metadata["year"],
                "duration_min": metadata["duration_min"],
                "rating": metadata["rating"],
                "genres": metadata["genres"],
                "overview": metadata["overview"],
                "similarity": round(similarity, 4),
            })
        return movies

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        pairs = [
            (query, f"{c['title']} ({c['year']}). {c['overview']}")
            for c in candidates
        ]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

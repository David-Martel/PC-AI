# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0",
#     "psycopg[binary]",
#     "pgvector",
#     "requests"
# ]
# ///
import argparse
import sys
from typing import List, Optional

import psycopg
import requests
from pgvector.psycopg import register_vector
from pydantic import BaseModel

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "nomic-embed-text"


class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: str


class OllamaEmbeddingResponse(BaseModel):
    embedding: List[float]


def get_embedding(
    text: str, model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL
) -> Optional[List[float]]:
    try:
        req_data = OllamaEmbeddingRequest(model=model, prompt=text)
        response = requests.post(url, json=req_data.model_dump())
        response.raise_for_status()
        resp_data = OllamaEmbeddingResponse.model_validate(response.json())
        return resp_data.embedding
    except Exception as e:
        print(f"Error fetching query embedding: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Query PC_AI semantic RAG database for similar codebase components using semantic embeddings."
    )
    parser.add_argument(
        "query",
        help="Text description or code snippet to search for across the repository",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="PostgreSQL connection URI (e.g. postgresql://pcai_user:pcai_password@localhost:5432/semantic_rag)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of deduplication results to return",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama embedding model to query (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL"
    )

    args = parser.parse_args()

    try:
        conn = psycopg.connect(args.db, autocommit=True)
        register_vector(conn)
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    emb = get_embedding(args.query, args.model, args.ollama_url)
    if not emb:
        print("Failed to compute embedding for query.")
        sys.exit(1)

    with conn.cursor() as cur:
        # pgvector cosine distance operator is <=>
        # We calculate similarity as 1 - distance
        cur.execute(
            """
            SELECT file_path, name, chunk_type, 1 - (embedding <=> %s::vector) AS similarity
            FROM code_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """,
            (emb, emb, args.limit),
        )

        results = cur.fetchall()

        print(
            f"\n--- Top {min(args.limit, len(results))} Semantic Overlap Matches ---\n"
        )
        if not results:
            print(
                "No matching code chunks found. Ensure the ingestion pipeline has been run."
            )

        for r in results:
            filepath, name, chunk_type, similarity = r[0], r[1], r[2], r[3]
            print(f"[{similarity:.3f}] {name} ({chunk_type})")
            print(f"      -> {filepath}\n")

    conn.close()


if __name__ == "__main__":
    main()

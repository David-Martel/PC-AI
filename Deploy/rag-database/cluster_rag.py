# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "psycopg[binary]",
#     "pgvector",
#     "pydantic>=2.0"
# ]
# ///
import argparse
import sys
from typing import List

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel


class DuplicationCluster(BaseModel):
    source_name: str
    source_file: str
    target_name: str
    target_file: str
    chunk_type: str
    similarity: float


def main():
    parser = argparse.ArgumentParser(
        description="Identify codebase-wide duplication clusters using pgvector self-joins."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="PostgreSQL connection URI (e.g. postgresql://pcai_user:changeme@localhost/semantic_rag)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.93,
        help="Cosine similarity threshold for duplication targeting (default: 0.93)",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max clusters to return")
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI integration mode (clean exit on DB fail, strict exit code 2 on duplication)",
    )

    args = parser.parse_args()

    try:
        conn = psycopg.connect(args.db, autocommit=True)
        register_vector(conn)
    except Exception as e:
        if getattr(args, "ci_mode", False):
            print(f"RAG DB not available in CI mode. Skipping check. ({e})")
            sys.exit(0)
        else:
            print(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)

    print(
        f"Scanning database for duplication clusters with similarity > {args.threshold}...\n"
    )

    clusters: List[DuplicationCluster] = []

    with conn.cursor() as cur:
        # Self-join to find similar chunks
        cur.execute(
            """
            SELECT
                a.name AS source_name, a.file_path AS source_file,
                b.name AS target_name, b.file_path AS target_file,
                a.chunk_type,
                1 - (a.embedding <=> b.embedding) AS similarity
            FROM code_chunks a
            JOIN code_chunks b ON a.id < b.id AND a.chunk_type = b.chunk_type
            WHERE 1 - (a.embedding <=> b.embedding) > %s
            ORDER BY similarity DESC
            LIMIT %s;
        """,
            (args.threshold, args.limit),
        )

        results = cur.fetchall()
        for r in results:
            clusters.append(
                DuplicationCluster(
                    source_name=r[0],
                    source_file=r[1],
                    target_name=r[2],
                    target_file=r[3],
                    chunk_type=r[4],
                    similarity=r[5],
                )
            )

    if not clusters:
        print(
            "No duplication clusters found above the threshold. Clean codebase! (Or check if AST ingestion succeeded)"
        )
    else:
        print(f"--- Top {len(clusters)} Semantic Duplication Clusters ---\n")
        for idx, cluster in enumerate(clusters, 1):
            print(
                f"Cluster #{idx} [{cluster.similarity:.3f} Match] - Type: {cluster.chunk_type}"
            )
            print(f"  A: {cluster.source_file} -> {cluster.source_name}")
            print(f"  B: {cluster.target_file} -> {cluster.target_name}\n")

        if getattr(args, "ci_mode", False):
            sys.exit(2)

    conn.close()


if __name__ == "__main__":
    main()

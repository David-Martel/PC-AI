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
import hashlib
import json
import subprocess
import sys
from typing import Dict, List, Optional

import psycopg
import requests
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field, ValidationError

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "nomic-embed-text"


class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: str


class OllamaEmbeddingResponse(BaseModel):
    embedding: List[float]


class AstGrepMetaNode(BaseModel):
    text: str


class AstGrepMetaVariables(BaseModel):
    single: Dict[str, AstGrepMetaNode] = Field(default_factory=dict)


class AstGrepMatch(BaseModel):
    text: str
    file: str
    metaVariables: Optional[AstGrepMetaVariables] = None


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
        print(f"Error fetching embedding from {url} for chunk: {e}", file=sys.stderr)
        return None


def ingest_chunk(
    db_conn: psycopg.Connection,
    filepath: str,
    chunk_text: str,
    chunk_type: str,
    name: str,
    model: str,
    url: str,
) -> str:
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM code_chunks WHERE content_hash = %s", (content_hash,)
        )
        if cur.fetchone():
            return "skipped"

    embedding = get_embedding(chunk_text, model, url)
    if not embedding:
        return "failed"

    with db_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO code_chunks (file_path, chunk_type, name, content, embedding, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path, name, chunk_type)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                content_hash = EXCLUDED.content_hash,
                created_at = now();
        """,
            (filepath, chunk_type, name, chunk_text, embedding, content_hash),
        )
    return "inserted"


def extract_with_sg(pattern: str, lang: str, cwd: str = ".") -> List[AstGrepMatch]:
    cmd = ["sg", "run", "-p", pattern, "-l", lang, "--json"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, encoding="utf-8"
        )
        if not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        return [AstGrepMatch.model_validate(item) for item in data]
    except ValidationError as ve:
        print(
            f"Pydantic validation error for AST-Grep output ({lang}): {ve}",
            file=sys.stderr,
        )
        return []
    except Exception as e:
        print(f"Error running ast-grep for {lang}: {e}", file=sys.stderr)
        return []


def process_language(
    conn: psycopg.Connection,
    lang: str,
    pattern: str,
    chunk_type: str,
    args: argparse.Namespace,
):
    print(f"Scanning {lang} files for {chunk_type}...")
    matches = extract_with_sg(pattern, lang, cwd=args.repo_dir)

    inserted = skipped = failed = 0

    for m in matches:
        text = m.text
        filepath = m.file

        name = f"unnamed_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        if m.metaVariables and "NAME" in m.metaVariables.single:
            name = m.metaVariables.single["NAME"].text

        if not text or not filepath:
            continue

        status = ingest_chunk(
            conn, filepath, text, chunk_type, name, args.model, args.ollama_url
        )
        if status == "inserted":
            inserted += 1
            print(f"  [+] Ingested {chunk_type}: {name} from {filepath}")
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    print(
        f"Finished {lang}: {inserted} inserted, {skipped} skipped, {failed} failed.\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PC_AI codebase semantic representations into PostgreSQL pgvector natively using ast-grep and ollama."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="PostgreSQL connection URI (e.g. postgresql://pcai_user:pcai_password@localhost:5432/semantic_rag)",
    )
    parser.add_argument(
        "--repo-dir", default=".", help="Root directory of the repository to scan."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama embedding model to use (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL"
    )

    args = parser.parse_args()

    try:
        conn = psycopg.connect(args.db, autocommit=True)
        register_vector(conn)
    except Exception as e:
        print(
            f"Failed to connect to PostgreSQL: {e}\nEnsure your PostgreSQL server is running natively and pgvector is installed."
        )
        sys.exit(1)

    print("Connected to database successfully. Beginning ingestion...\n")

    extractions = [
        {
            "lang": "rust",
            "pattern": "fn $NAME($$$ARGS) { $$$BODY }",
            "chunk_type": "rust_function",
        },
        {
            "lang": "rust",
            "pattern": "impl $NAME { $$$BODY }",
            "chunk_type": "rust_impl",
        },
        {
            "lang": "csharp",
            "pattern": "class $NAME { $$$BODY }",
            "chunk_type": "csharp_class",
        },
        {
            "lang": "csharp",
            "pattern": "public $MODIFIER $NAME($$$ARGS) { $$$BODY }",
            "chunk_type": "csharp_method",
        },
        {
            "lang": "bash",
            "pattern": "function $NAME { $$$BODY }",
            "chunk_type": "powershell_function",
        },
    ]

    for ext in extractions:
        process_language(conn, ext["lang"], ext["pattern"], ext["chunk_type"], args)

    print("Ingestion pipeline complete.")
    conn.close()


if __name__ == "__main__":
    main()

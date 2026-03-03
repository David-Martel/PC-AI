CREATE EXTENSION IF NOT EXISTS vector;

-- Enforce semantic deduplication for overlapping RAG ingestion runs
CREATE TABLE IF NOT EXISTS code_chunks (
    id bigserial PRIMARY KEY,
    file_path text NOT NULL,
    chunk_type text NOT NULL, -- e.g., 'powershell_function', 'rust_impl', 'csharp_class'
    name text,                -- The name of the symbol (function/class) to prevent duplicates
    content text NOT NULL,    -- The raw code text
    metadata jsonb,           -- Extracted synopses, parameters, parameters doc coverage
    embedding vector,         -- Open dimension size (native vLLM vs Ollama interop)
    content_hash text UNIQUE, -- SHA-256 hash or simple duplication marker
    created_at timestamptz DEFAULT now(),
    UNIQUE(file_path, name, chunk_type) -- Allow overwrite/upsert of specific symbols seamlessly
);

-- Index for semantic similarity using Cosine Distance
CREATE INDEX ON code_chunks USING hnsw (embedding vector_cosine_ops);

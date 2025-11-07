-- Supabase pgvector setup for CallGPT RAG application
-- Run this in your Supabase SQL Editor

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create documents table
-- IMPORTANT: Adjust vector dimension (384) to match your embedding model:
-- - sentence-transformers/all-MiniLM-L6-v2 → 384
-- - sentence-transformers/all-mpnet-base-v2 → 768
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(384),  -- ⚠️ CHANGE THIS if using a different model
  created_at timestamp with time zone default now()
);

-- 3. Create index for fast similarity search
create index if not exists documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Create RPC function for vector similarity search
-- This function is called by the CustomSupabaseVectorStore
create or replace function match_documents (
  query_embedding vector(384),  -- ⚠️ Must match table dimension
  match_count int default 10
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 5. (Optional) Enable Row Level Security if needed
-- alter table documents enable row level security;
-- create policy "Enable read access for all users" on documents for select using (true);
-- create policy "Enable insert for authenticated users only" on documents for insert with check (auth.role() = 'authenticated');

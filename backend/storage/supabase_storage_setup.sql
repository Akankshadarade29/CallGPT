-- Supabase Storage metadata table setup for CallGPT
-- Follow the layout: upload file to Storage and persist metadata in Postgres

-- 1. Create table to store uploaded file metadata
create table if not exists file_metadata (
  id uuid primary key default gen_random_uuid(),
  bucket_name text not null,
  object_name text not null,
  size bigint,
  content_type text,
  etag text,
  last_modified timestamp with time zone,
  public_url text,
  created_at timestamp with time zone default now(),
  unique(bucket_name, object_name)
);

-- 2. RLS Policies for Storage (works for ANY bucket)
-- These policies allow public access to all storage buckets
-- If you need per-bucket restrictions, modify the policies with bucket_id checks

-- Allow public uploads (INSERT) to any bucket
create policy "Allow public uploads to all buckets"
on storage.objects
for insert
with check (true);

-- Allow public reads (SELECT) from any bucket
create policy "Allow public reads from all buckets"
on storage.objects
for select
using (true);

-- Allow public updates (UPDATE) to any bucket (needed for upsert)
create policy "Allow public updates to all buckets"
on storage.objects
for update
using (true)
with check (true);

-- Allow public deletes (DELETE) from any bucket (needed for upsert)
create policy "Allow public deletes from all buckets"
on storage.objects
for delete
using (true);

-- Optional: If you want to restrict to specific bucket(s), replace 'true' with:
-- bucket_id = 'your-bucket-name'
-- Or for multiple buckets:
-- bucket_id in ('bucket1', 'bucket2', 'bucket3')

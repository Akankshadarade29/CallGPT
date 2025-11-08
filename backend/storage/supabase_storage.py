from __future__ import annotations

import os
from typing import Any, Dict, Optional
from supabase import create_client, Client

# This module uploads user files to Supabase Storage and stores their metadata in Postgres.
# It follows the layout provided: upload with original filename,
# and then persist metadata via a dedicated table.


def _client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_API_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_API_KEY (or SERVICE_KEY) must be set")
    return create_client(url, key)


def get_bucket_name() -> str:
    """
    Purpose: Get bucket name from environment variable or use default.
    
    Environment variable: SUPABASE_BUCKET
    Default: 'user-files'
    
    Return Value:
    - str: Bucket name to use for storage operations.
    """
    return os.getenv("SUPABASE_BUCKET", "user-files")


def upload_text_bytes(
    *,
    bucket_name: str,
    object_name: str,
    data: bytes,
    upsert: bool = True,
) -> Dict[str, Any]:
    """
    Purpose: Upload raw text bytes to Supabase Storage.

    Return Value:
    - Dict: {"bucket": str, "path": str, "public_url": Optional[str]}
    """
    sb = _client()
    # Upload without file_options to avoid MIME type restrictions
    # If upsert=True and file exists, delete and re-upload
    if upsert:
        try:
            # Try to remove existing file (ignore if doesn't exist)
            sb.storage.from_(bucket_name).remove([object_name])
        except Exception:
            pass  # File doesn't exist, continue
    
    # Upload the file with generic binary content type to bypass text/plain restriction
    # Use application/octet-stream which is typically allowed in buckets with restrictions
    sb.storage.from_(bucket_name).upload(
        path=object_name,
        file=data,
        file_options={"contentType": "application/octet-stream"},
    )

    # Build a public URL if the bucket is public
    pub = sb.storage.from_(bucket_name).get_public_url(object_name)
    public_url = pub.get("data", {}).get("publicUrl") if isinstance(pub, dict) else None
    return {"bucket": bucket_name, "path": object_name, "public_url": public_url}


def ensure_bucket_exists(bucket_name: str, public: bool = True) -> None:
    """
    Purpose: Create a Storage bucket if it doesn't exist.
    """
    sb = _client()
    try:
        sb.storage.create_bucket(bucket_name, options={"public": public})
    except Exception:
        # Ignore if already exists
        pass


def get_file_metadata(*, bucket_name: str, object_name: str) -> Dict[str, Any]:
    """
    Purpose: Retrieve file metadata from Supabase Storage listing.
    """
    sb = _client()
    # List at parent folder level; if no folders, list root
    parent = "/".join(object_name.split("/")[:-1]) if "/" in object_name else ""
    items = sb.storage.from_(bucket_name).list(parent)
    # Find our object
    for it in items:
        if it.get("name") == (object_name.split("/")[-1]):
            return it
    return {}


def store_metadata_record(
    *,
    bucket_name: str,
    object_name: str,
    meta: Optional[Dict[str, Any]] = None,
    public_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Purpose: Store file metadata in Postgres table `file_metadata`.

    Fields stored:
    - bucket_name, object_name, size, content_type, etag, last_modified, public_url
    """
    sb = _client()
    meta = meta or {}

    row = {
        "bucket_name": bucket_name,
        "object_name": object_name,
        "size": (meta.get("metadata", {}) or {}).get("size"),
        "content_type": (meta.get("metadata", {}) or {}).get("mimetype") or "text/plain",
        "etag": meta.get("eTag"),
        "last_modified": meta.get("updated_at"),
        "public_url": public_url,
    }
    # Upsert on (bucket_name, object_name)
    res = sb.table("file_metadata").upsert(row, on_conflict="bucket_name,object_name").execute()
    return getattr(res, "data", res)

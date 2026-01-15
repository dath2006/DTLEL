from fastapi import APIRouter
from app.engine import plagiarism_engine
from typing import List, Dict, Any

router = APIRouter()

@router.get("/documents")
async def get_documents():
    """
    Retrieves a list of unique documents currently in the Data Pool.
    Aggregates chunks by 'source' filename.
    """
    # Group by source
    docs: Dict[str, Any] = {}
    
    for idx, meta in plagiarism_engine.metadata_store.items():
        source = meta.get("source", "Unknown")
        if source not in docs:
            docs[source] = {
                "filename": source,
                "chunk_count": 0,
                "upload_timestamp": "N/A" # Simple MVP, could store time in meta later
            }
        docs[source]["chunk_count"] += 1

    return list(docs.values())

@router.get("/documents/{filename}/chunks")
async def get_document_chunks(filename: str):
    """
    Retrieves all chunks associated with a specific filename.
    """
    chunks = []
    for idx, meta in plagiarism_engine.metadata_store.items():
        if meta.get("source") == filename:
            chunks.append({
                "id": idx,
                "metadata": meta
            })
    return chunks
@router.delete("/documents/clear")
async def clear_data_pool():
    """
    Clears the entire Data Pool (Index + Metadata).
    """
    plagiarism_engine.clear_index()
    return {"message": "Data Pool cleared successfully"}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Deletes a specific document from the Data Pool.
    """
    full_filename = filename # Decode? FastAPI handles URL decoding usually.
    count = plagiarism_engine.delete_document(full_filename)
    return {"message": f"Deleted document '{filename}'", "chunks_removed": count}

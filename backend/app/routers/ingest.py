from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.utils.text_extract import extract_text_from_file
from app.utils.sentence_chunking import sentence_splitter
from app.engine import plagiarism_engine
from app.models import IngestResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingests a document into the plagiarism reference database.
    Now uses Sentence-Level Chunking for granular paraphrase detection.
    """
    try:
        text = await extract_text_from_file(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty document")
        
        # Split into sentences for granular indexing
        sentences = sentence_splitter.split_sentences(text)
        
        if not sentences:
             raise HTTPException(status_code=400, detail="Could not extract sentences")

        chunks = [s[0] for s in sentences]
        
        # Prepare metadata for each chunk
        metadatas = []
        for i, (sent_text, start, end) in enumerate(sentences):
            metadatas.append({
                "source": file.filename,
                "chunk_index": i,
                "text_snippet": sent_text[:100] + "...",
                "text": sent_text,
                "start_char": start,
                "end_char": end
            })
            
        # Add to index (Synchronous for now, can be backgrounded if large)
        plagiarism_engine.add_to_index(chunks, metadatas)
        
        return IngestResponse(
            id=plagiarism_engine.current_id, # return last ID
            message=f"Successfully ingested {file.filename}",
            chunk_count=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

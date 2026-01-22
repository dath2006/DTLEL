"""
Text Extraction Utilities (v2.0 - Async Streaming)

Optimizations:
- Streaming file reads to prevent memory exhaustion
- Size limits to prevent DoS attacks
- Graceful handling of encoding issues
"""

import io
from fastapi import UploadFile, HTTPException
import pypdf
import docx

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 64 * 1024  # 64KB chunks


async def _stream_file_content(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> bytes:
    """
    Stream file content in chunks to prevent memory exhaustion.
    Raises HTTPException if file exceeds max_size.
    """
    chunks = []
    total_size = 0
    
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB"
            )
        chunks.append(chunk)
    
    return b"".join(chunks)


async def extract_text_from_file(file: UploadFile) -> str:
    """
    Extracts text from an uploaded file (PDF, DOCX, TXT).
    Uses streaming to handle large files gracefully.
    """
    content_type = file.content_type or ""
    filename = (file.filename or "").lower()
    
    try:
        if "pdf" in content_type or filename.endswith(".pdf"):
            return await _read_pdf(file)
        elif "word" in content_type or "officedocument" in content_type or filename.endswith(".docx"):
            return await _read_docx(file)
        elif "text" in content_type or filename.endswith(".txt"):
            return await _read_txt(file)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


async def _read_pdf(file: UploadFile) -> str:
    """Extract text from PDF with streaming."""
    content = await _stream_file_content(file)
    pdf_reader = pypdf.PdfReader(io.BytesIO(content))
    text_parts = []
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_parts.append(extracted)
    return "\n".join(text_parts)


async def _read_docx(file: UploadFile) -> str:
    """Extract text from DOCX with streaming."""
    content = await _stream_file_content(file)
    doc = docx.Document(io.BytesIO(content))
    return "\n".join([para.text for para in doc.paragraphs])


async def _read_txt(file: UploadFile) -> str:
    """Extract text from TXT with streaming and encoding fallback."""
    content = await _stream_file_content(file)
    # Try UTF-8 first, fallback to latin-1 (covers most cases)
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="replace")


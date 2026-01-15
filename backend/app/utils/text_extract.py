import io
from fastapi import UploadFile, HTTPException
import pypdf
import docx

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Extracts text from an uploaded file (PDF, DOCX, TXT).
    """
    content_type = file.content_type or ""
    filename = file.filename.lower()
    
    try:
        if "pdf" in content_type or filename.endswith(".pdf"):
            return await _read_pdf(file)
        elif "word" in content_type or "officedocument" in content_type or filename.endswith(".docx"):
            return await _read_docx(file)
        elif "text" in content_type or filename.endswith(".txt"):
            return await _read_txt(file)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

async def _read_pdf(file: UploadFile) -> str:
    content = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(content))
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

async def _read_docx(file: UploadFile) -> str:
    content = await file.read()
    doc = docx.Document(io.BytesIO(content))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

async def _read_txt(file: UploadFile) -> str:
    content = await file.read()
    return content.decode("utf-8")

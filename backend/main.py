from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

app = FastAPI(title="Hybrid AI & Plagiarism Detection System", version="1.0.0")

# CORS setup (Allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000","https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import ingest, analyze, data
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(analyze.router, prefix="/api/v1", tags=["Analyze"])
app.include_router(data.router, prefix="/api/v1", tags=["Data Pool"])

@app.get("/health")
def health_check():
    return {"status": "ok", "app_name": settings.APP_NAME}

@app.get("/")
def root():
    return {"message": "Welcome to the Hybrid Integrity System API 's"}

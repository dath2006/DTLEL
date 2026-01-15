import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Hybrid Integrity System"
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration
    PLAGIARISM_MODEL_NAME: str = "all-mpnet-base-v2"
    AI_DETECTOR_MODEL_NAME: str = "PirateXX/AI-Content-Detector"
    
    # Device Configuration
    @property
    def DEVICE(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"

settings = Settings()

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    ENV: str = Field(default="development", env="ENV")
    PROJECT_NAME: str = "Cyber Threat Detection API"
    API_V1_STR: str = "/api/v1"
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Model Artifact paths
    MODEL_PATH: Path = Field(default=Path("models/model.pkl"), env="MODEL_PATH")
    SCALER_PATH: Path = Field(default=Path("models/scaler.pkl"), env="SCALER_PATH")
    FEATURE_COLS_PATH: Path = Field(default=Path("models/feature_columns.json"), env="FEATURE_COLS_PATH")
    ONNX_MODEL_PATH: Path = Field(default=Path("models/model.onnx"), env="ONNX_MODEL_PATH")
    
    # Database Settings
    POSTGRES_SERVER: str = Field(default="localhost", env="POSTGRES_SERVER")
    POSTGRES_USER: str = Field(default="postgres", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="postgres", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="threat_db", env="POSTGRES_DB")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    DATABASE_URL: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Security Settings
    SECRET_KEY: str = Field(default="supersecretsecuritykey-change-in-production", env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()

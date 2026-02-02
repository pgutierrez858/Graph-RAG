from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_embedding_model: str = "nomic-embed-text"

    # Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
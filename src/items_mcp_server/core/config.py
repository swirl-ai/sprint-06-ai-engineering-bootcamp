from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    QDRANT_URL: str = 'qdrant'
    QDRANT_COLLECTION_NAME_ITEMS: str = "Amazon-items-collection-02-items"
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str

    model_config = SettingsConfigDict(env_file=".env")


config = Config()
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    QDRANT_URL: str
    QDRANT_COLLECTION_NAME_ITEMS: str = "Amazon-items-collection-02-items"
    QDRANT_COLLECTION_NAME_REVIEWS: str = "Amazon-items-collection-02-reviews"
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_PROVIDER: str
    GENERATION_MODEL: str
    GENERATION_MODEL_PROVIDER: str
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    COORDINATOR_AGENT_PROMPT_TEMPLATE_PATH: str = "src/api/rag/prompts/coordinator_agent.yaml"
    PRODUCT_QA_AGENT_PROMPT_TEMPLATE_PATH: str = "src/api/rag/prompts/product_qa_agent.yaml"
    SHOPPING_CART_AGENT_PROMPT_TEMPLATE_PATH: str = "src/api/rag/prompts/shopping_cart_agent.yaml"
    POSTGRES_CONN_STRING: str = "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"

    model_config = SettingsConfigDict(env_file=".env")

class Settings(BaseSettings):

    DEFAULT_TIMEOUT: float = 30.0
    VERSION: str = "0.1.0"

config = Config()
settings = Settings()
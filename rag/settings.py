from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://localhost/k8s_docs_rag"
    voyage_api_key: str | None = None
    # Hard cap on cumulative billed Voyage tokens, tracked in data/usage/voyage.jsonl.
    # The account's free allowance is 200M; stay far below it.
    voyage_token_budget: int = 20_000_000
    anthropic_api_key: str | None = None
    # Hard cap on cumulative Claude API spend, tracked in data/usage/anthropic.jsonl.
    anthropic_budget_usd: float = 5.0
    # Let the API's /ask call Claude on this key. Off by default: the demo serves /examples.
    live_answers: bool = False


def get_settings() -> Settings:
    return Settings()

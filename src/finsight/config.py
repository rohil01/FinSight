"""Application settings loaded from environment / .env via pydantic-settings."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration in one place. Values come from the
    environment or a `.env` file; every field can be overridden per-run."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow the `model_name` field
    )

    # API keys (validated at startup by `validate_keys`)
    nvidia_api_key: str = Field(default="", description="NVIDIA API key for hosted NIMs")
    tavily_api_key: str = Field(default="", description="Tavily search API key")

    # Model / agent behaviour. Both defaults are NVIDIA NIMs on the API catalog.
    model_name: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description="NVIDIA NIM model id for planning, research, critique, and writing",
    )
    fast_model_name: str = Field(
        default="nvidia/nemotron-3-nano-30b-a3b",
        description="Smaller NVIDIA NIM for domain classification and eval judging",
    )
    max_iterations: int = Field(default=3, ge=1, description="Max plan→research→critique loops")
    max_search_results: int = Field(default=3, ge=1, description="Tavily results per sub-question")
    scrape_char_limit: int = Field(default=2000, ge=100, description="Max chars kept from a scraped page")

    # Estimated USD per 1M tokens. Override to match the model you actually run.
    price_input_per_million: float = Field(default=1.25, ge=0)
    price_output_per_million: float = Field(default=10.0, ge=0)

    def validate_keys(self) -> None:
        """Fail fast with a clear message if required API keys are missing."""
        missing = []
        if not self.nvidia_api_key:
            missing.append("NVIDIA_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required API key(s): {', '.join(missing)}. "
                "Set them in your environment or in a .env file "
                "(see .env.example)."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()

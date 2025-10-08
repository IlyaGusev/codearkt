from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PLANNING_LAST_N: int = 50
    PLANNING_CONTENT_MAX_LENGTH: int = 4000
    DEFAULT_MAX_ITERATIONS: int = 20
    MAX_LENGTH_TRUNCATE_CONTENT: int = 20000

    AGENT_TOOL_PREFIX: str = "agent__"

    DEFAULT_SERVER_HOST: str = "0.0.0.0"
    DEFAULT_SERVER_PORT: int = 5055

    # Executor settings
    CODEARKT_EXECUTOR_URL: Optional[str] = None
    EXECUTOR_IMAGE: str = (
        "phoenix120/codearkt_http@sha256:e00d11db4bc70918f61ebd53e19b0b2f382af6165346322af401b701118404e1"
    )
    DOCKER_MEM_LIMIT: str = "1g"
    DOCKER_CPU_QUOTA: int = 50000
    DOCKER_CPU_PERIOD: int = 100000
    DOCKER_CLEANUP_TIMEOUT: int = 10
    DOCKER_PIDS_LIMIT: int = 256
    DOCKER_NET_NAME: str = "codearkt_sandbox_net"

    # Timeout settings
    EVENT_BUS_STREAM_TIMEOUT: int = 24 * 60 * 60
    EXEC_TIMEOUT: int = 24 * 60 * 60
    PROXY_SSE_READ_TIMEOUT: int = 12 * 60 * 60

    # OpenRouter settings
    BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = ""
    HTTP_REFERRER: str = "https://github.com/IlyaGusev/codearkt/"
    X_TITLE: str = "CodeArkt"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    @field_validator("OPENROUTER_API_KEY")  # type: ignore
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("OPENROUTER_API_KEY must not be empty")
        return v


settings = Settings()

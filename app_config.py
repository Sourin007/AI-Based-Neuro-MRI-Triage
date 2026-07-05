"""API-layer configuration (separate from the ML pipeline's brain_tumor_ai.Settings).

Holds web/serving concerns — CORS, upload limits, rate limiting, logging, error
tracking — and validates them at startup so the process fails fast with a clear
message instead of misbehaving in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class AppConfig:
    environment: str
    cors_allow_origins: tuple[str, ...]
    max_upload_bytes: int
    analyze_rate_limit: str
    sentry_dsn: str | None
    log_level: str
    json_logs: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        environment = os.getenv("ENVIRONMENT", "development").strip().lower()

        origins = _split_csv(os.getenv("CORS_ALLOW_ORIGINS", ""))
        if not origins and environment != "production":
            # Convenience default for local dev (Vite dev server).
            origins = ("http://localhost:5173", "http://127.0.0.1:5173")

        try:
            max_upload_mb = float(os.getenv("MAX_UPLOAD_MB", "15"))
        except ValueError as exc:
            raise RuntimeError("MAX_UPLOAD_MB must be a number.") from exc

        config = cls(
            environment=environment,
            cors_allow_origins=origins,
            max_upload_bytes=int(max_upload_mb * 1024 * 1024),
            analyze_rate_limit=os.getenv("ANALYZE_RATE_LIMIT", "10/minute"),
            sentry_dsn=os.getenv("SENTRY_DSN") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            json_logs=os.getenv("JSON_LOGS", "true").strip().lower() in {"1", "true", "yes"},
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.max_upload_bytes <= 0:
            raise RuntimeError("MAX_UPLOAD_MB must be greater than 0.")

        if self.environment == "production":
            if not self.cors_allow_origins:
                raise RuntimeError(
                    "CORS_ALLOW_ORIGINS must be set to your frontend origin(s) in production "
                    "(comma-separated), e.g. https://your-app.vercel.app"
                )
            if "*" in self.cors_allow_origins:
                raise RuntimeError(
                    "CORS_ALLOW_ORIGINS must not be '*' in production. "
                    "List explicit frontend origins instead."
                )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

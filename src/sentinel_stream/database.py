import logging
import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
logger = logging.getLogger("sentinel_stream.database")

DATABASE_URL = os.getenv("SUPABASE_DB_URL", "").strip()

# If DATABASE_URL is missing or placeholder, provide graceful fallback
if not DATABASE_URL or "[PROJECT_REF]" in DATABASE_URL:
    logger.warning("SUPABASE_DB_URL is unset or contains placeholders. Falling back to local SQLite audit database.")
    DATABASE_URL = "sqlite:///./sentinel_audit.db"

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _get_utc_now():
    return datetime.now(UTC)


class TransactionAudit(Base):
    __tablename__ = "transaction_audits"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    merchant = Column(String, nullable=False)
    fraud_score = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    decision = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=_get_utc_now)


def init_db():
    """Ensure database schema tables are created."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.error("Failed to initialize database tables: %s", exc)


def get_db():
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

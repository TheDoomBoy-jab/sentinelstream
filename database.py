import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("SUPABASE_DB_URL", "")
if not DATABASE_URL:
    raise ValueError("SUPABASE_DB_URL env variable missing")
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
class TransactionAudit(Base):
    __tablename__ = "transaction_audits"
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    merchant = Column(String, nullable=False)
    fraud_score = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    decision = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
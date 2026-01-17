"""
Cached Analysis Model - Store expensive analysis results
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from app.core.database import Base

class CachedAnalysis(Base):
    __tablename__ = "cached_analyses"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False, index=True)

    # Type of analysis (e.g., "competitor_analysis_123" where 123 is competitor_id)
    type = Column(String(100), nullable=False, index=True)

    # The actual cached data (JSON)
    data = Column(JSON, nullable=False)

    # Expiration and timestamps
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def is_valid(self) -> bool:
        """Check if cache is still valid"""
        return datetime.utcnow() < self.expires_at

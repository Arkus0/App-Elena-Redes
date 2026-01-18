from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class ViralBenchmark(Base):
    """
    Benchmark data for viral content analysis.
    Stores historical post performance classified as viral or flop
    to serve as reference points for the clustering algorithm.
    """
    __tablename__ = "viral_benchmarks"

    id = Column(Integer, primary_key=True, index=True)
    niche = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Scraped data reference
    platform_post_id = Column(String, index=True)
    platform = Column(String, nullable=False)

    # Performance metrics
    engagement_real = Column(Float, nullable=False)  # Normalized engagement score
    is_viral = Column(Boolean, nullable=False)  # True = High performing, False = Low performing

    # Feature vector for clustering (normalized values stored as JSON)
    # Includes: hook_energy, retention, cut_rate, semantic_pca/umap, etc.
    features_json = Column(JSON, nullable=False)

    def __repr__(self):
        return f"<ViralBenchmark(id={self.id}, niche='{self.niche}', viral={self.is_viral})>"

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from datetime import datetime
from app.core.database import Base

class MLTrainingSample(Base):
    __tablename__ = "ml_training_queue"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String(255), index=True, nullable=True)

    # Feature vector for training (multimodal + tabular)
    features = Column(JSON, nullable=False)

    # Actual performance targets (likes, shares, etc.)
    actual_metrics = Column(JSON, nullable=False)

    # Performance delta (predicted vs actual)
    delta_score = Column(Float, nullable=False)

    # Training priority
    priority = Column(Float, default=0.0)
    is_high_priority = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

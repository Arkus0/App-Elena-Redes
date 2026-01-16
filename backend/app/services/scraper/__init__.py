"""
Scraper module - Balanced sampling to fix Survivor Bias

This module implements a balanced data collection strategy that captures both:
- Top performing content (viral posts)
- Bottom performing content (flops)

This enables the ML model to learn discriminative features for both success and failure.
"""

from .balanced_scraper import BalancedScraper, SamplingStrategy
from .preprocessing import (
    DataPreprocessor,
    calculate_engagement_ratio,
    label_viral_status,
    balance_dataset,
)

__all__ = [
    "BalancedScraper",
    "SamplingStrategy",
    "DataPreprocessor",
    "calculate_engagement_ratio",
    "label_viral_status",
    "balance_dataset",
]

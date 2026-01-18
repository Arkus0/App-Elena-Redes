"""
BrandPulse AI - ML Module
==========================

This module contains scripts for training and managing engagement prediction models.

Scripts:
--------
- pretrain_base_model.py: Train base model on synthetic data for cold start mitigation
- train.py: Train niche-specific models with cold start handling (fine-tuning)

Usage:
------
1. First, pretrain the base model:
   python ml/pretrain_base_model.py

2. Then, train niche-specific models:
   python ml/train.py --niche restaurante --data-file data.csv

Cold Start Strategy:
-------------------
- If niche has >= 300 samples: Train from scratch
- If niche has < 300 samples: Fine-tune from base model
"""

__version__ = "1.0.0"

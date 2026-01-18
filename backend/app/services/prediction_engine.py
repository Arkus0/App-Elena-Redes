"""
Prediction Engine - 100% Data-Driven Recommendation System
==========================================================

Implements 'Viral Cluster Recommendation' using KMeans clustering on
benchmarked performance data.

Algorithm:
1. Load ViralBenchmark data for a specific niche.
2. Separate into 'Viral' (High Engagement) and 'Flop' (Low Engagement) clusters.
3. Train KMeans on both subsets to find centroids (archetypes).
4. For candidate content, calculate Euclidean distance to nearest Viral vs Flop centroid.
5. Score = (Distance_to_Flop - Distance_to_Viral) / (Distance_to_Flop + Distance_to_Viral)
   - Score > 0: Closer to Viral
   - Score < 0: Closer to Flop

Hybrid Integration:
- Combines Clustering Score (70%) with XGBoost Prediction (30%) for final ranking.

Author: BrandPulse AI
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

from app.models.viral_benchmark import ViralBenchmark
from app.services.growth_prediction_engine import get_growth_prediction_engine

logger = logging.getLogger(__name__)

class ViralClusterer:
    """
    Clustering engine to identify viral archetypes and score content candidates.
    Optimized for small datasets (<100 samples) common in niche SMBs.
    """

    # Features used for clustering (must match AnalyticsEngine output)
    CLUSTER_FEATURES = [
        "hook_energy",
        "hook_cut_rate",
        "face_in_hook",
        "production_quality_score",
        "visual_energy",
        "tempo",
        "brightness_variance"
    ]

    # Semantic features (will be appended)
    SEMANTIC_FEATURES = [f"sem_umap_{i}" for i in range(1, 11)]

    def __init__(self, niche: str):
        self.niche = niche
        self.scaler = StandardScaler()
        self.kmeans_viral: Optional[KMeans] = None
        self.kmeans_flop: Optional[KMeans] = None
        self.viral_centroids = None
        self.flop_centroids = None
        self.is_fitted = False
        self.feature_keys = self.CLUSTER_FEATURES + self.SEMANTIC_FEATURES

    async def load_and_fit(self, db: AsyncSession):
        """
        Load benchmarks from DB and fit clusters.
        """
        # Fetch benchmarks
        query = select(ViralBenchmark).where(ViralBenchmark.niche == self.niche)
        result = await db.execute(query)
        benchmarks = result.scalars().all()

        if not benchmarks or len(benchmarks) < 10:
            logger.warning(f"Insufficient benchmarks for niche '{self.niche}' (found {len(benchmarks)}). Clustering skipped.")
            self.is_fitted = False
            return

        # Prepare data
        viral_samples = []
        flop_samples = []

        for b in benchmarks:
            feats = b.features_json
            # Extract ordered feature vector
            vector = [float(feats.get(k, 0.0)) for k in self.feature_keys]

            if b.is_viral:
                viral_samples.append(vector)
            else:
                flop_samples.append(vector)

        if not viral_samples or not flop_samples:
            logger.warning("Need both viral and flop samples to fit clusters.")
            self.is_fitted = False
            return

        # Stack and Scale
        X_all = np.vstack(viral_samples + flop_samples)
        self.scaler.fit(X_all)

        X_viral = self.scaler.transform(np.vstack(viral_samples))
        X_flop = self.scaler.transform(np.vstack(flop_samples))

        # Fit KMeans
        # n_clusters is small due to expected low data volume per niche
        n_clusters_viral = min(5, len(viral_samples))
        n_clusters_flop = min(5, len(flop_samples))

        self.kmeans_viral = KMeans(n_clusters=n_clusters_viral, random_state=42, n_init=10)
        self.kmeans_viral.fit(X_viral)
        self.viral_centroids = self.kmeans_viral.cluster_centers_

        self.kmeans_flop = KMeans(n_clusters=n_clusters_flop, random_state=42, n_init=10)
        self.kmeans_flop.fit(X_flop)
        self.flop_centroids = self.kmeans_flop.cluster_centers_

        self.is_fitted = True
        logger.info(f"ViralClusterer fitted for {self.niche}: {len(viral_samples)} viral, {len(flop_samples)} flop samples.")

    def predict_viral_score(self, candidate_features: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate viral proximity score (-1 to 1).
        Score > 0 means closer to Viral centroids.

        Returns:
            Tuple(score, reason)
        """
        if not self.is_fitted:
            return 0.0, "Not enough data for clustering comparison"

        # Prepare vector
        vector = [float(candidate_features.get(k, 0.0)) for k in self.feature_keys]
        X_cand = self.scaler.transform([vector])

        # Distances to nearest centroids
        # min(euclidean_distances) returns distance to closest cluster center
        dist_viral = np.min(euclidean_distances(X_cand, self.viral_centroids))
        dist_flop = np.min(euclidean_distances(X_cand, self.flop_centroids))

        # Normalized score: (Flop - Viral) / (Flop + Viral)
        # If dist_viral is 0 (exact match), score is 1.
        # If dist_flop is 0, score is -1.
        denom = dist_flop + dist_viral + 1e-6
        score = (dist_flop - dist_viral) / denom

        # Interpret reason
        if score > 0.5:
            match_idx = np.argmin(euclidean_distances(X_cand, self.viral_centroids))
            reason = "Alta similitud con arquetipos virales (Match Viral)"
        elif score < -0.5:
            reason = "Alta similitud con contenido de bajo rendimiento (Riesgo Flop)"
        else:
            reason = "Zona neutra: características mixtas"

        return float(score), reason

    def explain_score(self, candidate_features: Dict[str, Any]) -> List[str]:
        """Generate data-driven explanation for the score."""
        if not self.is_fitted:
            return ["Faltan datos de benchmark para explicar."]

        vector = [float(candidate_features.get(k, 0.0)) for k in self.feature_keys]
        X_cand = self.scaler.transform([vector])[0]

        # Find nearest viral centroid
        dists = euclidean_distances([X_cand], self.viral_centroids)
        nearest_idx = np.argmin(dists)
        viral_center = self.viral_centroids[nearest_idx]

        # Compare features to centroid
        # Identify features with smallest difference (contributors)
        # and largest difference (detractors)
        diffs = np.abs(X_cand - viral_center)

        # Get feature importance based on proximity to ideal
        explanations = []

        # Check specific key features
        idx_hook = self.feature_keys.index("hook_energy")
        if abs(X_cand[idx_hook] - viral_center[idx_hook]) < 0.5:
            explanations.append("Hook Energy alineado con virales")

        idx_prod = self.feature_keys.index("production_quality_score")
        if X_cand[idx_prod] > viral_center[idx_prod]:
            explanations.append("Calidad de producción superior al promedio viral")

        return explanations

class PredictionEngine:
    """
    Orchestrator for 100% Data-Driven Recommendations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.growth_engine = get_growth_prediction_engine()

    async def recommend_best_candidate(
        self,
        candidates: List[Dict[str, Any]],
        niche: str
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates using Hybrid Score.

        Formula:
        Final Score = 0.7 * Cluster_Score (Viral Match) + 0.3 * ML_Score (XGBoost)

        Args:
            candidates: List of feature dicts from AnalyticsEngine
            niche: Business niche (e.g., 'real_estate')

        Returns:
            Ranked list with scores and reasons.
        """
        # 1. Initialize Clusterer
        clusterer = ViralClusterer(niche)
        await clusterer.load_and_fit(self.db)

        ranked_results = []

        for cand in candidates:
            # A. Cluster Score (-1 to 1)
            cluster_score, cluster_reason = clusterer.predict_viral_score(cand)

            # B. ML Prediction (0 to 100) -> Normalize to 0-1
            # We assume candidates have features ready for XGBoost
            # GrowthEngine expects slightly different keys, we map loosely
            # Assuming 'cand' has keys from AnalyticsEngine
            try:
                ml_result = self.growth_engine.predict_with_explanation(cand, allow_cold_start=True)
                raw_ml_score = ml_result.predicted_rpi_score
                # Normalize ML score (approx 0-5 scale log-transformed typically, normalize to 0-1)
                ml_score_norm = min(1.0, max(0.0, raw_ml_score / 5.0))
                ml_explanation = ml_result.explanation_text
            except Exception as e:
                logger.error(f"ML Prediction failed: {e}")
                ml_score_norm = 0.5
                ml_explanation = "ML Error"

            # C. Hybrid Score
            # Map cluster score from [-1, 1] to [0, 1] for weighted avg
            cluster_score_norm = (cluster_score + 1) / 2

            # Weight: 70% Historic Pattern (Cluster), 30% Predictive Model
            final_score = (0.7 * cluster_score_norm) + (0.3 * ml_score_norm)

            ranked_results.append({
                "candidate_id": cand.get("source", "unknown"),
                "final_score": round(final_score * 100, 1), # 0-100 scale
                "viral_match_score": round(cluster_score, 2),
                "predicted_engagement": round(ml_score_norm * 100, 1),
                "reason": cluster_reason,
                "data_driven_insights": clusterer.explain_score(cand),
                "ml_explanation": ml_explanation,
                "features": cand
            })

        # Sort by Final Score DESC
        ranked_results.sort(key=lambda x: x["final_score"], reverse=True)

        return ranked_results

# Factory
def get_prediction_engine(db: AsyncSession) -> PredictionEngine:
    return PredictionEngine(db)

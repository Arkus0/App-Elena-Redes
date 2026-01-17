"""
ML Service - Hybrid ML/LLM Architecture for Cost-Efficient Predictions
Uses XGBoost/RandomForest for fast predictions, LLM only for creative generation

FEEDBACK LOOP ARCHITECTURE (Human-in-the-Loop Reinforcement Learning):
======================================================================
The model learns from real-world performance by:
1. Storing predictions before publication
2. Collecting actual performance metrics after 24-48 hours
3. Calculating delta between predicted and actual engagement
4. Flagging high-delta samples (>20% difference) as high priority
5. Incorporating these samples in the next training cycle

This allows the model to learn from its mistakes and continuously improve.
"""
import os
import re
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import shap
import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# FEEDBACK LOOP DATA STRUCTURES
# =============================================================================

# Threshold for flagging samples as high priority training data
# If actual differs from predicted by more than this percentage, it's flagged
HIGH_PRIORITY_DELTA_THRESHOLD = 20.0  # 20% difference

# Minimum hours after posting to collect performance data
MIN_HOURS_FOR_FEEDBACK = 24

# Maximum hours after posting (older data may not be relevant)
MAX_HOURS_FOR_FEEDBACK = 168  # 7 days


@dataclass
class PerformanceFeedback:
    """
    Feedback data structure for ML training loop.

    Captures the delta between predicted and actual performance,
    along with metadata about why this sample is valuable for training.
    """
    content_id: int
    predicted_score: float
    actual_score: float
    delta_percent: float
    is_high_priority: bool
    training_priority: float
    metrics: Dict[str, Any]
    collected_at: datetime
    analysis_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "predicted_score": round(self.predicted_score, 2),
            "actual_score": round(self.actual_score, 2),
            "delta_percent": round(self.delta_percent, 2),
            "is_high_priority": self.is_high_priority,
            "training_priority": round(self.training_priority, 4),
            "metrics": self.metrics,
            "collected_at": self.collected_at.isoformat(),
            "analysis_notes": self.analysis_notes,
        }


@dataclass
class TrainingQueueItem:
    """
    Item in the training queue for the next retraining cycle.
    """
    content_id: int
    features: Dict[str, Any]
    actual_engagement: float
    priority: float
    added_at: datetime
    feedback: PerformanceFeedback

    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to format suitable for model training."""
        sample = self.features.copy()
        sample["engagement_score"] = self.actual_engagement
        sample["_priority"] = self.priority
        sample["_source"] = "feedback_loop"
        return sample

# Model storage directory
MODEL_DIR = Path("./ml_models")
MODEL_DIR.mkdir(exist_ok=True)


class FeatureExtractor:
    """
    Extract features from content for ML predictions
    Features are designed for local SMB social media content
    """

    # Emoji patterns
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )

    # Engagement trigger words (Spanish focused for local SMBs)
    TRIGGER_WORDS = {
        "question": ["?", "cuál", "qué", "cómo", "por qué", "quién", "dónde", "cuándo"],
        "urgency": ["ahora", "hoy", "último", "limitado", "exclusivo", "ya", "rápido"],
        "social_proof": ["clientes", "testimonios", "opiniones", "reviews", "valoraciones"],
        "value": ["gratis", "regalo", "descuento", "oferta", "promoción", "ahorra"],
        "curiosity": ["secreto", "descubre", "sorpresa", "increíble", "no creerás"],
        "action": ["comenta", "guarda", "comparte", "sígueme", "dm", "escríbeme", "haz click"],
        "emotion": ["amor", "feliz", "alegría", "pasión", "sueño", "gracias"],
        "transformation": ["antes", "después", "transformación", "cambio", "resultado"],
    }

    # Hook patterns
    HOOK_PATTERNS = {
        "pov": r"pov[:\s]|punto de vista",
        "question": r"^[¿?]|^\w+\s*\?",
        "number": r"^\d+\s+\w+|top\s*\d+|\d+\s*(cosas|tips|errores|razones)",
        "bold_claim": r"nunca|siempre|todo|nadie|el mejor|el peor|imposible",
        "story": r"historia|storytime|cuando|un día|me pasó",
        "how_to": r"cómo\s+\w+|aprende\s+a|tutorial|paso\s+a\s+paso",
        "reveal": r"secreto|te cuento|descubre|te revelo|no sabías",
    }

    # CTA patterns
    CTA_PATTERNS = {
        "comment": r"comenta|cuéntame|opina|dime|escribe",
        "save": r"guarda|guardar|guárdalo|save",
        "share": r"comparte|compartir|etiqueta|tag",
        "follow": r"sígueme|sigue|follow|seguir",
        "dm": r"dm|mensaje|escríbeme|mensaje directo",
        "link": r"link|enlace|bio|click|pincha",
    }

    @classmethod
    def extract_features(cls, content: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features from content
        Returns feature dict for ML model input
        """
        caption = content.get("caption", "") or ""
        caption_lower = caption.lower()

        features = {}

        # === Text Length Features ===
        features["caption_length"] = len(caption)
        features["caption_words"] = len(caption.split())
        features["caption_lines"] = caption.count("\n") + 1
        features["avg_word_length"] = (
            np.mean([len(w) for w in caption.split()]) if caption.split() else 0
        )

        # === Emoji Features ===
        emojis = cls.EMOJI_PATTERN.findall(caption)
        features["emoji_count"] = len(emojis)
        features["emoji_density"] = len(emojis) / max(len(caption), 1) * 100

        # === Hashtag Features ===
        hashtags = content.get("hashtags", []) or re.findall(r"#\w+", caption)
        features["hashtag_count"] = len(hashtags)
        features["hashtag_density"] = len(hashtags) / max(features["caption_words"], 1)

        # === Mention Features ===
        mentions = content.get("mentions", []) or re.findall(r"@\w+", caption)
        features["mention_count"] = len(mentions)

        # === Trigger Word Features ===
        for trigger_type, words in cls.TRIGGER_WORDS.items():
            count = sum(1 for word in words if word in caption_lower)
            features[f"trigger_{trigger_type}"] = count

        # === Hook Detection ===
        first_line = caption.split("\n")[0] if caption else ""
        first_line_lower = first_line.lower()

        for hook_type, pattern in cls.HOOK_PATTERNS.items():
            features[f"hook_{hook_type}"] = 1 if re.search(pattern, first_line_lower, re.IGNORECASE) else 0

        # === CTA Detection ===
        for cta_type, pattern in cls.CTA_PATTERNS.items():
            features[f"cta_{cta_type}"] = 1 if re.search(pattern, caption_lower, re.IGNORECASE) else 0

        # Total CTAs
        features["cta_count"] = sum(features[f"cta_{t}"] for t in cls.CTA_PATTERNS.keys())

        # === Format Features ===
        content_format = content.get("content_format", content.get("type", "unknown"))
        features["is_reel"] = 1 if content_format in ["reel", "tiktok_video", "video"] else 0
        features["is_carousel"] = 1 if content_format == "carousel" else 0
        features["is_static"] = 1 if content_format in ["static_image", "static", "image"] else 0

        # === Video Features ===
        duration = content.get("video_duration_seconds", content.get("video_duration", 0)) or 0
        features["video_duration"] = duration
        features["video_optimal_length"] = 1 if 15 <= duration <= 60 else 0

        # === Audio Features ===
        audio = content.get("audio_name", content.get("recommended_audio", ""))
        features["has_audio"] = 1 if audio else 0
        features["is_trending_audio"] = 1 if audio and "original" not in audio.lower() else 0

        # === Timing Features ===
        posted_at = content.get("posted_at")
        if posted_at:
            try:
                if isinstance(posted_at, str):
                    dt = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                else:
                    dt = posted_at
                features["hour_of_day"] = dt.hour
                features["day_of_week"] = dt.weekday()
                features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
                features["is_prime_time"] = 1 if dt.hour in [11, 12, 13, 19, 20, 21] else 0
            except:
                features["hour_of_day"] = 12
                features["day_of_week"] = 2
                features["is_weekend"] = 0
                features["is_prime_time"] = 1
        else:
            features["hour_of_day"] = 12
            features["day_of_week"] = 2
            features["is_weekend"] = 0
            features["is_prime_time"] = 1

        # === Engagement (for training, 0 for prediction) ===
        features["likes"] = content.get("likes_count", content.get("likes", 0)) or 0
        features["comments"] = content.get("comments_count", content.get("comments", 0)) or 0
        features["shares"] = content.get("shares_count", content.get("shares", 0)) or 0
        features["saves"] = content.get("saves_count", content.get("saves", 0)) or 0
        features["views"] = content.get("views_count", content.get("video_views", content.get("plays", 0))) or 0

        # === Business Type (will be encoded) ===
        features["business_type"] = content.get("business_type", "otros")

        return features

    @classmethod
    def extract_batch(cls, contents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features for multiple content items"""
        features_list = [cls.extract_features(c) for c in contents]
        return pd.DataFrame(features_list)


class MLPredictor:
    """
    ML Predictor with XGBoost/RandomForest models
    Provides engagement scoring, format recommendation, and trigger suggestions
    """

    FEATURE_COLUMNS = [
        "caption_length", "caption_words", "caption_lines", "avg_word_length",
        "emoji_count", "emoji_density", "hashtag_count", "hashtag_density",
        "mention_count",
        "trigger_question", "trigger_urgency", "trigger_social_proof",
        "trigger_value", "trigger_curiosity", "trigger_action",
        "trigger_emotion", "trigger_transformation",
        "hook_pov", "hook_question", "hook_number", "hook_bold_claim",
        "hook_story", "hook_how_to", "hook_reveal",
        "cta_comment", "cta_save", "cta_share", "cta_follow", "cta_dm", "cta_link",
        "cta_count",
        "is_reel", "is_carousel", "is_static",
        "video_duration", "video_optimal_length",
        "has_audio", "is_trending_audio",
        "hour_of_day", "day_of_week", "is_weekend", "is_prime_time",
        "business_type_encoded",
    ]

    FORMAT_CLASSES = ["reel", "carousel", "static_image", "tiktok_video"]

    def __init__(self):
        self.engagement_model: Optional[xgb.XGBRegressor] = None
        self.format_model: Optional[RandomForestClassifier] = None
        self.trigger_model: Optional[xgb.XGBClassifier] = None
        self.business_type_encoder = LabelEncoder()
        self.format_encoder = LabelEncoder()
        self.shap_explainer_engagement = None
        self.shap_explainer_format = None
        self.is_trained = False

        # Try to load existing models
        self._load_models()

    def _get_model_path(self, name: str) -> Path:
        """Get path for a model file"""
        return MODEL_DIR / f"{name}.joblib"

    def _load_models(self):
        """Load trained models from disk"""
        try:
            engagement_path = self._get_model_path("engagement_model")
            format_path = self._get_model_path("format_model")
            trigger_path = self._get_model_path("trigger_model")
            encoder_path = self._get_model_path("encoders")

            if all(p.exists() for p in [engagement_path, format_path, encoder_path]):
                self.engagement_model = joblib.load(engagement_path)
                self.format_model = joblib.load(format_path)

                if trigger_path.exists():
                    self.trigger_model = joblib.load(trigger_path)

                encoders = joblib.load(encoder_path)
                self.business_type_encoder = encoders["business_type"]
                self.format_encoder = encoders["format"]

                self.is_trained = True
                logger.info("Loaded trained ML models from disk")

                # Initialize SHAP explainers
                self._init_shap_explainers()

        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.is_trained = False

    def _save_models(self):
        """Save trained models to disk"""
        try:
            joblib.dump(self.engagement_model, self._get_model_path("engagement_model"))
            joblib.dump(self.format_model, self._get_model_path("format_model"))

            if self.trigger_model:
                joblib.dump(self.trigger_model, self._get_model_path("trigger_model"))

            encoders = {
                "business_type": self.business_type_encoder,
                "format": self.format_encoder,
            }
            joblib.dump(encoders, self._get_model_path("encoders"))

            logger.info("Saved ML models to disk")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def _init_shap_explainers(self):
        """Initialize SHAP explainers for model interpretability"""
        try:
            if self.engagement_model:
                self.shap_explainer_engagement = shap.TreeExplainer(self.engagement_model)
            if self.format_model:
                self.shap_explainer_format = shap.TreeExplainer(self.format_model)
            logger.info("SHAP explainers initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainers: {e}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input"""
        df = df.copy()

        # Encode business type
        if "business_type" in df.columns:
            # Handle unseen categories
            known_types = set(self.business_type_encoder.classes_) if hasattr(self.business_type_encoder, 'classes_') else set()
            df["business_type_encoded"] = df["business_type"].apply(
                lambda x: self.business_type_encoder.transform([x])[0] if x in known_types else 0
            )
        else:
            df["business_type_encoded"] = 0

        # Select only needed columns
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        return df[feature_cols].fillna(0)

    def train(self, training_data: List[Dict[str, Any]], retrain: bool = False):
        """
        Train all ML models on provided data
        training_data: List of content dicts with engagement metrics
        """
        if self.is_trained and not retrain:
            logger.info("Models already trained. Use retrain=True to force retraining.")
            return

        logger.info(f"Training ML models on {len(training_data)} samples...")

        # Extract features
        df = FeatureExtractor.extract_batch(training_data)

        # Fit encoders
        business_types = df["business_type"].unique().tolist()
        if "otros" not in business_types:
            business_types.append("otros")
        self.business_type_encoder.fit(business_types)

        self.format_encoder.fit(self.FORMAT_CLASSES)

        # Prepare features
        df["business_type_encoded"] = self.business_type_encoder.transform(df["business_type"])

        # Calculate engagement score (target for regression)
        df["engagement_score"] = (
            df["likes"] +
            df["comments"] * 3 +
            df["saves"] * 5 +
            df["shares"] * 4
        )
        # Normalize to 0-100 scale
        max_engagement = df["engagement_score"].quantile(0.95)
        df["engagement_score_normalized"] = (df["engagement_score"] / max(max_engagement, 1) * 100).clip(0, 100)

        # Determine best format (target for classification)
        format_cols = ["is_reel", "is_carousel", "is_static"]
        df["best_format"] = df[format_cols].idxmax(axis=1).str.replace("is_", "")
        df.loc[df["is_reel"] == 1, "best_format"] = "reel"

        # Features for training
        X = self._prepare_features(df)

        # === Train Engagement Model (XGBoost Regression) ===
        y_engagement = df["engagement_score_normalized"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_engagement, test_size=0.2, random_state=42
        )

        self.engagement_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.engagement_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.engagement_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"Engagement Model RMSE: {rmse:.2f}")

        # === Train Format Recommendation Model (Random Forest) ===
        # For format, we want to recommend based on content features
        # Use high-engagement posts to learn what formats work best
        high_engagement_mask = df["engagement_score_normalized"] > df["engagement_score_normalized"].median()
        X_format = self._prepare_features(df[high_engagement_mask])
        y_format = df.loc[high_engagement_mask, "best_format"]

        if len(y_format.unique()) > 1:
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                X_format, y_format, test_size=0.2, random_state=42
            )

            self.format_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            self.format_model.fit(X_train_f, y_train_f)

            # Evaluate
            y_pred_f = self.format_model.predict(X_test_f)
            accuracy = accuracy_score(y_test_f, y_pred_f)
            logger.info(f"Format Model Accuracy: {accuracy:.2%}")
        else:
            # Default model if not enough variety
            self.format_model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.format_model.fit(X_format, ["reel"] * len(X_format))

        # === Train Trigger Suggestion Model (Multi-label) ===
        # Predict which triggers lead to high engagement
        trigger_cols = [c for c in df.columns if c.startswith("trigger_")]
        high_eng_triggers = df.loc[high_engagement_mask, trigger_cols].mean()
        best_triggers = high_eng_triggers.nlargest(3).index.tolist()

        # Store for recommendations
        self._best_triggers = best_triggers

        self.is_trained = True
        self._save_models()
        self._init_shap_explainers()

        logger.info("ML models trained and saved successfully")

    def predict_engagement(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict engagement score for content
        Returns score and SHAP explanation
        """
        if not self.is_trained:
            # Return mock prediction if not trained
            return self._mock_engagement_prediction(content)

        features = FeatureExtractor.extract_features(content)
        df = pd.DataFrame([features])
        X = self._prepare_features(df)

        # Predict
        score = float(self.engagement_model.predict(X)[0])
        score = max(0, min(100, score))  # Clip to 0-100

        # SHAP explanation
        explanation = self._get_shap_explanation(X, "engagement")

        return {
            "score": round(score, 1),
            "confidence": self._calculate_confidence(X),
            "explanation": explanation,
            "feature_importance": self._get_feature_importance("engagement"),
        }

    def recommend_format(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend best content format
        """
        if not self.is_trained:
            return self._mock_format_recommendation(content)

        features = FeatureExtractor.extract_features(content)
        df = pd.DataFrame([features])
        X = self._prepare_features(df)

        # Predict probabilities
        probs = self.format_model.predict_proba(X)[0]
        classes = self.format_model.classes_

        # Get top recommendations
        recommendations = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "recommended_format": recommendations[0][0],
            "confidence": float(recommendations[0][1] * 100),
            "alternatives": [
                {"format": fmt, "score": float(prob * 100)}
                for fmt, prob in recommendations[1:3]
            ],
            "explanation": self._get_shap_explanation(X, "format"),
        }

    def suggest_triggers(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest engagement triggers to include
        """
        features = FeatureExtractor.extract_features(content)

        # Analyze current triggers
        current_triggers = {
            k.replace("trigger_", ""): v
            for k, v in features.items()
            if k.startswith("trigger_") and v > 0
        }

        # Missing high-value triggers
        all_triggers = list(FeatureExtractor.TRIGGER_WORDS.keys())
        missing_triggers = [t for t in all_triggers if t not in current_triggers]

        # Prioritize based on trained model or defaults
        if hasattr(self, "_best_triggers"):
            priority_triggers = [t.replace("trigger_", "") for t in self._best_triggers if t.replace("trigger_", "") in missing_triggers]
        else:
            priority_triggers = ["question", "action", "curiosity"]

        suggestions = []
        for trigger in priority_triggers[:3]:
            examples = FeatureExtractor.TRIGGER_WORDS.get(trigger, [])[:3]
            suggestions.append({
                "trigger_type": trigger,
                "impact": "high" if trigger in ["question", "action"] else "medium",
                "examples": examples,
                "reason": self._get_trigger_reason(trigger),
            })

        return {
            "current_triggers": current_triggers,
            "suggestions": suggestions,
            "improvement_potential": self._calculate_improvement_potential(features),
        }

    def get_full_prediction(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get complete ML prediction with all components
        This is called before LLM generation for the hybrid approach
        """
        engagement = self.predict_engagement(content)
        format_rec = self.recommend_format(content)
        triggers = self.suggest_triggers(content)

        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(content, engagement, format_rec, triggers)

        return {
            "engagement_prediction": engagement,
            "format_recommendation": format_rec,
            "trigger_suggestions": triggers,
            "optimization_suggestions": suggestions,
            "ml_summary": self._generate_ml_summary(engagement, format_rec, triggers),
        }

    def _get_shap_explanation(self, X: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Generate SHAP-based explanation"""
        try:
            if model_type == "engagement" and self.shap_explainer_engagement:
                shap_values = self.shap_explainer_engagement.shap_values(X)

                # Get feature importances from SHAP
                feature_names = X.columns.tolist()
                importances = np.abs(shap_values[0])

                # Top positive and negative factors
                sorted_idx = np.argsort(shap_values[0])

                top_positive = [
                    {"feature": feature_names[i], "impact": float(shap_values[0][i])}
                    for i in sorted_idx[-3:][::-1] if shap_values[0][i] > 0
                ]

                top_negative = [
                    {"feature": feature_names[i], "impact": float(shap_values[0][i])}
                    for i in sorted_idx[:3] if shap_values[0][i] < 0
                ]

                # Human-readable explanation
                explanation_text = self._shap_to_text(top_positive, top_negative)

                return {
                    "top_positive_factors": top_positive,
                    "top_negative_factors": top_negative,
                    "explanation_text": explanation_text,
                }

            elif model_type == "format" and self.shap_explainer_format:
                shap_values = self.shap_explainer_format.shap_values(X)
                return {"note": "Format recommendation based on similar high-engagement content"}

        except Exception as e:
            logger.warning(f"SHAP explanation error: {e}")

        return {"note": "Explanation not available"}

    def _shap_to_text(self, positive: List[Dict], negative: List[Dict]) -> str:
        """Convert SHAP values to human-readable text"""
        feature_descriptions = {
            "emoji_count": "uso de emojis",
            "emoji_density": "densidad de emojis",
            "hashtag_count": "número de hashtags",
            "cta_count": "llamadas a la acción",
            "cta_comment": "CTA de comentarios",
            "cta_save": "CTA de guardado",
            "hook_question": "hook de pregunta",
            "hook_pov": "formato POV",
            "trigger_question": "preguntas en el texto",
            "trigger_action": "palabras de acción",
            "is_reel": "formato Reel",
            "is_carousel": "formato Carousel",
            "video_optimal_length": "duración óptima del video",
            "is_prime_time": "horario prime",
            "caption_length": "longitud del caption",
        }

        parts = []

        if positive:
            pos_factors = [
                feature_descriptions.get(p["feature"], p["feature"])
                for p in positive[:2]
            ]
            parts.append(f"Factores positivos: {', '.join(pos_factors)}")

        if negative:
            neg_factors = [
                feature_descriptions.get(n["feature"], n["feature"])
                for n in negative[:2]
            ]
            parts.append(f"Áreas de mejora: {', '.join(neg_factors)}")

        return ". ".join(parts) if parts else "Análisis basado en patrones de contenido exitoso"

    def _get_feature_importance(self, model_type: str) -> List[Dict[str, Any]]:
        """Get global feature importance"""
        try:
            if model_type == "engagement" and self.engagement_model:
                importances = self.engagement_model.feature_importances_
                feature_names = self.FEATURE_COLUMNS[:len(importances)]

                sorted_idx = np.argsort(importances)[::-1][:10]
                return [
                    {"feature": feature_names[i], "importance": float(importances[i])}
                    for i in sorted_idx
                ]
        except:
            pass
        return []

    def _calculate_confidence(self, X: pd.DataFrame) -> float:
        """Calculate prediction confidence based on feature coverage"""
        non_zero = (X.iloc[0] != 0).sum()
        total = len(X.columns)
        return min(95, max(60, (non_zero / total) * 100))

    def _calculate_improvement_potential(self, features: Dict) -> float:
        """Calculate how much the content could improve with suggestions"""
        potential = 0

        # Missing CTAs
        if features.get("cta_count", 0) == 0:
            potential += 20

        # No question hook
        if features.get("hook_question", 0) == 0 and features.get("trigger_question", 0) == 0:
            potential += 15

        # No emojis
        if features.get("emoji_count", 0) == 0:
            potential += 10

        # Suboptimal video length
        if features.get("is_reel", 0) == 1 and features.get("video_optimal_length", 0) == 0:
            potential += 10

        return min(potential, 50)

    def _get_trigger_reason(self, trigger: str) -> str:
        """Get reason why a trigger is recommended"""
        reasons = {
            "question": "Las preguntas aumentan comentarios un 150%+ al invitar respuestas",
            "action": "Los CTAs claros multiplican la interacción directa",
            "curiosity": "La curiosidad mantiene la atención y aumenta visualizaciones completas",
            "urgency": "La urgencia impulsa acciones inmediatas",
            "transformation": "El contenido before/after tiene 3x más engagement",
            "emotion": "Las emociones crean conexión y aumentan compartidos",
            "value": "El valor percibido aumenta guardados",
            "social_proof": "La prueba social genera confianza y conversiones",
        }
        return reasons.get(trigger, "Mejora el engagement general")

    def _generate_optimization_suggestions(
        self,
        content: Dict,
        engagement: Dict,
        format_rec: Dict,
        triggers: Dict
    ) -> List[str]:
        """Generate specific optimization suggestions"""
        suggestions = []

        features = FeatureExtractor.extract_features(content)

        # Format suggestion
        current_format = "reel" if features.get("is_reel") else "carousel" if features.get("is_carousel") else "static"
        recommended = format_rec.get("recommended_format", "reel")

        if current_format != recommended and format_rec.get("confidence", 0) > 70:
            suggestions.append(f"Considera cambiar a formato {recommended} para este contenido")

        # CTA suggestions
        if features.get("cta_count", 0) == 0:
            suggestions.append("Añade un CTA claro (ej: '¿Cuál prefieres? Comenta 👇' o 'Guarda para después')")

        # Hook suggestions
        if not any(features.get(f"hook_{h}", 0) for h in ["question", "pov", "number"]):
            suggestions.append("Usa un hook más potente: pregunta, POV, o número en los primeros 3 segundos")

        # Emoji suggestions
        if features.get("emoji_count", 0) < 3:
            suggestions.append("Añade 3-5 emojis relevantes para mejorar el visual scanning")

        # Trigger suggestions
        for trigger in triggers.get("suggestions", [])[:2]:
            if trigger.get("impact") == "high":
                suggestions.append(f"Incluye elementos de {trigger['trigger_type']}: {trigger['examples'][0] if trigger.get('examples') else ''}")

        return suggestions[:5]

    def _generate_ml_summary(self, engagement: Dict, format_rec: Dict, triggers: Dict) -> str:
        """Generate summary text for the ML prediction"""
        score = engagement.get("score", 50)
        format_name = format_rec.get("recommended_format", "reel")
        improvement = triggers.get("improvement_potential", 0)

        if score >= 80:
            quality = "excelente"
        elif score >= 60:
            quality = "bueno"
        elif score >= 40:
            quality = "mejorable"
        else:
            quality = "necesita optimización"

        summary = f"Predicción ML: Score {score}/100 ({quality}). "
        summary += f"Formato recomendado: {format_name}. "

        if improvement > 20:
            summary += f"Potencial de mejora: +{improvement}% con las sugerencias."

        return summary

    # =========================================================================
    # FEEDBACK LOOP - Human-in-the-Loop Reinforcement Learning
    # =========================================================================

    def __init_feedback_queue(self):
        """Initialize the training queue if not exists."""
        if not hasattr(self, '_training_queue'):
            self._training_queue: List[TrainingQueueItem] = []
            self._feedback_history: List[PerformanceFeedback] = []

    def register_performance_feedback(
        self,
        content_id: int,
        metrics: Dict[str, Any],
        original_content: Optional[Dict[str, Any]] = None,
        predicted_score: Optional[float] = None
    ) -> PerformanceFeedback:
        """
        Register actual performance metrics for a published content piece.

        This is the core method of the Human-in-the-Loop feedback system.
        It compares predicted engagement against actual performance and
        flags high-delta samples for priority retraining.

        FLOW:
        1. Calculate actual engagement score from metrics
        2. Compare with predicted score
        3. If delta > 20%, flag as HIGH_PRIORITY training sample
        4. Add to training queue for next retraining cycle

        Args:
            content_id: ID of the GeneratedContent record
            metrics: Actual performance metrics from Instagram/TikTok
                     Expected keys: likes, comments, saves, shares, views,
                                   reach, impressions, retention_rate
            original_content: Optional content dict (for feature extraction)
            predicted_score: Optional predicted score (if not provided,
                            will try to look up from content)

        Returns:
            PerformanceFeedback with analysis results

        Raises:
            ValueError: If metrics are invalid or insufficient
        """
        self.__init_feedback_queue()

        # Validate metrics
        required_metrics = ["likes", "comments"]
        if not all(k in metrics for k in required_metrics):
            raise ValueError(f"Missing required metrics: {required_metrics}")

        # Calculate actual engagement score (same formula as prediction training)
        actual_score = self._calculate_engagement_score(metrics)

        # Get predicted score
        if predicted_score is None:
            # In a real implementation, this would query the database
            # For now, use a placeholder or the score from original_content
            if original_content and "engagement_score" in original_content:
                predicted_score = original_content.get("engagement_score", 50.0)
            else:
                predicted_score = 50.0  # Default if unknown
                logger.warning(f"No predicted score for content {content_id}, using default")

        # Calculate delta percentage
        if predicted_score > 0:
            delta_percent = ((actual_score - predicted_score) / predicted_score) * 100
        else:
            delta_percent = 100.0 if actual_score > 0 else 0.0

        # Determine if this is a high priority training sample
        is_high_priority = abs(delta_percent) > HIGH_PRIORITY_DELTA_THRESHOLD

        # Calculate training priority score
        training_priority = self._calculate_training_priority(
            delta_percent=delta_percent,
            metrics=metrics,
            content=original_content
        )

        # Generate analysis notes
        analysis_notes = self._generate_feedback_analysis(
            predicted=predicted_score,
            actual=actual_score,
            delta=delta_percent,
            metrics=metrics
        )

        # Create feedback object
        feedback = PerformanceFeedback(
            content_id=content_id,
            predicted_score=predicted_score,
            actual_score=actual_score,
            delta_percent=delta_percent,
            is_high_priority=is_high_priority,
            training_priority=training_priority,
            metrics=metrics,
            collected_at=datetime.utcnow(),
            analysis_notes=analysis_notes
        )

        # Store in history
        self._feedback_history.append(feedback)

        # If we have original content, add to training queue
        if original_content is not None:
            features = FeatureExtractor.extract_features(original_content)
            queue_item = TrainingQueueItem(
                content_id=content_id,
                features=features,
                actual_engagement=actual_score,
                priority=training_priority,
                added_at=datetime.utcnow(),
                feedback=feedback
            )
            self._training_queue.append(queue_item)

            if is_high_priority:
                logger.info(
                    f"HIGH PRIORITY training sample flagged: content_id={content_id}, "
                    f"delta={delta_percent:.1f}%, priority={training_priority:.3f}"
                )

        logger.info(
            f"Registered feedback for content {content_id}: "
            f"predicted={predicted_score:.1f}, actual={actual_score:.1f}, "
            f"delta={delta_percent:.1f}%, high_priority={is_high_priority}"
        )

        return feedback

    def _calculate_engagement_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate normalized engagement score from raw metrics.

        Uses same formula as training to ensure consistency:
        score = likes + comments*3 + saves*5 + shares*4

        Then normalizes to 0-100 scale.
        """
        likes = metrics.get("likes", 0) or 0
        comments = metrics.get("comments", 0) or 0
        saves = metrics.get("saves", 0) or 0
        shares = metrics.get("shares", 0) or 0
        views = metrics.get("views", 0) or 0

        # Weighted engagement sum
        raw_score = likes + (comments * 3) + (saves * 5) + (shares * 4)

        # Normalize based on views (if available) or absolute scale
        if views > 0:
            # Engagement rate based normalization
            engagement_rate = raw_score / views
            # Scale to 0-100 (typical engagement rates are 1-10%)
            normalized = min(100, engagement_rate * 1000)
        else:
            # Absolute scale normalization (assuming max ~10000 engagement)
            normalized = min(100, (raw_score / 100) * 10)

        return round(normalized, 2)

    def _calculate_training_priority(
        self,
        delta_percent: float,
        metrics: Dict[str, Any],
        content: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate training priority score for a feedback sample.

        Higher priority samples are more valuable for model learning:
        - High delta = model made a big mistake, needs correction
        - Diverse content types = helps model generalize
        - Recent data = more relevant to current trends

        Priority formula:
        priority = base_delta_score * diversity_multiplier * recency_multiplier

        Returns:
            Priority score (0-1, higher = more valuable)
        """
        # Base priority from delta magnitude
        # Larger errors are more valuable for learning
        base_priority = min(1.0, abs(delta_percent) / 100.0)

        # Boost for very high deltas (model was very wrong)
        if abs(delta_percent) > 50:
            base_priority *= 1.5
        elif abs(delta_percent) > HIGH_PRIORITY_DELTA_THRESHOLD:
            base_priority *= 1.2

        # Diversity bonus for underrepresented content types
        diversity_multiplier = 1.0
        if content:
            content_format = content.get("content_format", "")
            # Carousel and static are less common, boost their priority
            if content_format == "carousel":
                diversity_multiplier = 1.3
            elif content_format in ["static", "static_image"]:
                diversity_multiplier = 1.2

        # Engagement volume bonus (high-engagement content is more informative)
        views = metrics.get("views", 0) or metrics.get("reach", 0) or 0
        volume_multiplier = 1.0
        if views > 10000:
            volume_multiplier = 1.3
        elif views > 1000:
            volume_multiplier = 1.1

        # Combine factors
        priority = base_priority * diversity_multiplier * volume_multiplier

        # Clamp to 0-1
        return min(1.0, max(0.0, priority))

    def _generate_feedback_analysis(
        self,
        predicted: float,
        actual: float,
        delta: float,
        metrics: Dict[str, Any]
    ) -> str:
        """Generate human-readable analysis of the prediction error."""
        notes = []

        # Direction of error
        if delta > HIGH_PRIORITY_DELTA_THRESHOLD:
            notes.append(f"Model UNDERESTIMATED by {delta:.1f}%")
            notes.append("Content performed better than expected")
        elif delta < -HIGH_PRIORITY_DELTA_THRESHOLD:
            notes.append(f"Model OVERESTIMATED by {abs(delta):.1f}%")
            notes.append("Content underperformed expectations")
        else:
            notes.append(f"Prediction within acceptable range (delta: {delta:.1f}%)")

        # Analyze which metrics drove the difference
        likes = metrics.get("likes", 0)
        comments = metrics.get("comments", 0)
        saves = metrics.get("saves", 0)
        shares = metrics.get("shares", 0)

        if comments > likes * 0.1:
            notes.append("High comment ratio suggests strong audience connection")
        if saves > likes * 0.05:
            notes.append("High save ratio indicates valuable/educational content")
        if shares > likes * 0.03:
            notes.append("High share ratio shows viral potential")

        return "; ".join(notes)

    def get_training_queue(self, min_priority: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get queued training samples above minimum priority.

        Args:
            min_priority: Minimum priority score (0-1)

        Returns:
            List of training samples ready for the next training cycle
        """
        self.__init_feedback_queue()

        samples = [
            item.to_training_sample()
            for item in self._training_queue
            if item.priority >= min_priority
        ]

        # Sort by priority (highest first)
        samples.sort(key=lambda x: x.get("_priority", 0), reverse=True)

        return samples

    def get_high_priority_samples(self) -> List[Dict[str, Any]]:
        """Get only high priority training samples (>20% delta)."""
        self.__init_feedback_queue()

        return [
            item.to_training_sample()
            for item in self._training_queue
            if item.feedback.is_high_priority
        ]

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        self.__init_feedback_queue()

        if not self._feedback_history:
            return {
                "total_samples": 0,
                "high_priority_count": 0,
                "avg_delta_percent": 0,
                "model_bias": "unknown"
            }

        deltas = [f.delta_percent for f in self._feedback_history]
        high_priority = [f for f in self._feedback_history if f.is_high_priority]

        avg_delta = np.mean(deltas)

        # Determine if model has systematic bias
        if avg_delta > 10:
            bias = "underestimating"
        elif avg_delta < -10:
            bias = "overestimating"
        else:
            bias = "calibrated"

        return {
            "total_samples": len(self._feedback_history),
            "high_priority_count": len(high_priority),
            "queue_size": len(self._training_queue),
            "avg_delta_percent": round(avg_delta, 2),
            "delta_std": round(np.std(deltas), 2),
            "model_bias": bias,
            "underestimated_count": sum(1 for d in deltas if d > HIGH_PRIORITY_DELTA_THRESHOLD),
            "overestimated_count": sum(1 for d in deltas if d < -HIGH_PRIORITY_DELTA_THRESHOLD),
        }

    def clear_training_queue(self):
        """Clear the training queue after a training cycle."""
        self.__init_feedback_queue()
        cleared = len(self._training_queue)
        self._training_queue = []
        logger.info(f"Cleared {cleared} samples from training queue")
        return cleared

    def retrain_with_feedback(
        self,
        additional_data: Optional[List[Dict[str, Any]]] = None,
        min_samples: int = 30
    ) -> bool:
        """
        Retrain the model using accumulated feedback data.

        Combines high-priority feedback samples with any additional
        real data to improve model accuracy.

        Args:
            additional_data: Optional list of additional training samples
            min_samples: Minimum samples required for retraining

        Returns:
            True if retraining was successful, False if insufficient data
        """
        self.__init_feedback_queue()

        # Collect training data
        feedback_samples = self.get_training_queue(min_priority=0.3)

        all_samples = []
        all_samples.extend(feedback_samples)

        if additional_data:
            all_samples.extend(additional_data)

        if len(all_samples) < min_samples:
            logger.warning(
                f"Insufficient data for retraining: {len(all_samples)} samples "
                f"(minimum: {min_samples}). Collect more feedback."
            )
            return False

        logger.info(
            f"Retraining model with {len(all_samples)} samples "
            f"({len(feedback_samples)} from feedback loop)"
        )

        # Perform retraining
        self.train(all_samples, retrain=True)

        # Clear used samples from queue
        self.clear_training_queue()

        return True

    # === Mock predictions when model not trained ===

    def _mock_engagement_prediction(self, content: Dict) -> Dict[str, Any]:
        """Mock prediction for demo/testing"""
        features = FeatureExtractor.extract_features(content)

        # Simple heuristic score
        score = 50
        score += min(features.get("emoji_count", 0) * 2, 10)
        score += min(features.get("cta_count", 0) * 5, 15)
        score += features.get("hook_question", 0) * 10
        score += features.get("is_reel", 0) * 10
        score = min(max(score, 20), 95)

        return {
            "score": score,
            "confidence": 70,
            "explanation": {
                "explanation_text": "Predicción basada en heurísticas (modelo no entrenado). Entrena con datos reales para predicciones precisas.",
                "top_positive_factors": [{"feature": "cta_count", "impact": 5}] if features.get("cta_count", 0) > 0 else [],
                "top_negative_factors": [],
            },
            "feature_importance": [],
        }

    def _mock_format_recommendation(self, content: Dict) -> Dict[str, Any]:
        """Mock format recommendation"""
        return {
            "recommended_format": "reel",
            "confidence": 75,
            "alternatives": [
                {"format": "carousel", "score": 60},
                {"format": "static_image", "score": 40},
            ],
            "explanation": {"note": "Reels tienen el mayor alcance orgánico en 2026"},
        }


# ==========================================================================
# Global Instance & Factory
# ==========================================================================

# Global ML predictor instance
ml_predictor = MLPredictor()


def get_ml_predictor() -> MLPredictor:
    """Get the global ML predictor instance"""
    return ml_predictor


# Note: Synthetic data generation has been removed.
# The model now uses Cold Start heuristic prediction until sufficient
# real performance data is collected (minimum 30 samples).
# Use register_performance_feedback() to collect training data from
# actual content performance, then retrain_with_feedback() when ready.

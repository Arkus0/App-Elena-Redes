#!/usr/bin/env python3
"""
Semantic Hook Detection for Viral Engagement Prediction
=========================================================

Replaces naive RegEx-based hook detection with robust semantic similarity
using SentenceTransformer embeddings. Captures creative variations that
keyword matching misses.

APPROACH:
=========
1. Pre-define a corpus of 100+ known viral hooks (Spanish/Andaluz focus)
2. Generate embeddings for all base hooks (cached on first use)
3. For each new caption/transcript, compute cosine similarity vs all hooks
4. Hook score = max_similarity + bonuses for curiosity indicators

WHY SEMANTIC > REGEX:
=====================
- RegEx "cómo" misses "De qué manera", "El método para", "Así es como"
- RegEx "secreto" misses "Lo que nadie te dice", "Esto no lo sabías"
- Semantic catches creative variations: "POV: eres tu crush" vs "pov:"
- Works across Spanish dialects and creative spellings

VIRAL HOOK CATEGORIES:
======================
1. Question hooks - Invitan respuesta mental
2. POV hooks - Perspectiva inmersiva
3. Reveal/Secret hooks - Curiosidad por información oculta
4. Transformation hooks - Antes/después impactante
5. Number hooks - Listas y cuantificación
6. Bold claim hooks - Afirmaciones provocativas
7. Story hooks - Narrativa personal
8. How-to hooks - Valor educativo
9. Urgency hooks - FOMO y escasez
10. Social proof hooks - Validación social

Author: BrandPulse AI
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HOOK_EMBEDDINGS_CACHE_PATH = MODELS_DIR / "semantic_hooks_cache.pkl"

# Ensure directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Bonuses for additional curiosity indicators
QUESTION_BONUS = 0.05  # Contains ? or inverted ?
CURIOSITY_EMOJI_BONUS = 0.03  # Contains curiosity emojis
ELLIPSIS_BONUS = 0.02  # Contains ... (suspense)

# Similarity thresholds
HIGH_HOOK_THRESHOLD = 0.7  # Very strong hook similarity
MEDIUM_HOOK_THRESHOLD = 0.5  # Moderate hook similarity
LOW_HOOK_THRESHOLD = 0.3  # Weak hook similarity

# Curiosity emojis that boost engagement
CURIOSITY_EMOJIS = {"👀", "🤔", "😱", "🔥", "💡", "⚡", "🤯", "❓", "❗", "👇", "🧵", "📍", "🚨", "⬇️", "🎯"}


# =============================================================================
# Viral Hooks Database (100+ hooks in Spanish with Andalusian variations)
# =============================================================================

VIRAL_HOOKS_DATABASE = {
    # =========================================================================
    # QUESTION HOOKS - Invitan respuesta mental automática
    # =========================================================================
    "question": [
        # Direct questions
        "¿Sabías que esto existe?",
        "¿Por qué nadie habla de esto?",
        "¿Cuál prefieres?",
        "¿Qué harías en esta situación?",
        "¿Me creerías si te digo que...?",
        "¿Conoces este truco?",
        "¿Cómo es posible que no lo supieras?",
        "¿Te ha pasado esto alguna vez?",
        "¿Quién más necesita ver esto?",
        "¿Estás cometiendo este error?",
        "¿Cuántos de estos conocías?",
        "¿Sabes cuál es el secreto?",
        "¿Qué opinas de esto?",
        # Andalusian/informal variations
        "¿Pero esto qué es?",
        "¿Cómo mola esto no?",
        "¿Pero tú has visto esto?",
        "¿Cómo no sabía yo esto?",
    ],

    # =========================================================================
    # POV HOOKS - Perspectiva inmersiva (muy virales en Reels/TikTok)
    # =========================================================================
    "pov": [
        "POV: Descubres esto por primera vez",
        "POV: Eres tu propio jefe",
        "POV: Encuentras el lugar perfecto",
        "POV: Tu vida cambia con este truco",
        "POV: Llegas a tu café favorito",
        "POV: Te regalan esto",
        "POV: Pruebas esto por primera vez",
        "POV: Ves tu transformación",
        "Punto de vista: eres el cliente",
        "Tu perspectiva cuando descubres esto",
        "Imagina que esto te pasa a ti",
        "Así se siente cuando logras esto",
    ],

    # =========================================================================
    # REVEAL/SECRET HOOKS - Generan curiosidad por información exclusiva
    # =========================================================================
    "reveal": [
        "El secreto que nadie te cuenta",
        "Lo que no te dicen sobre esto",
        "Esto no lo sabías",
        "Te revelo el truco",
        "El secreto mejor guardado de",
        "Lo que los expertos no quieren que sepas",
        "Descubre lo que estás haciendo mal",
        "La verdad que nadie te dice",
        "Esto te va a cambiar la vida",
        "Lo que me hubiera gustado saber antes",
        "El método secreto para",
        "Aquí está la verdad sobre",
        "Te cuento algo que pocos saben",
        "Esto es lo que realmente funciona",
        "El truco que nadie te enseña",
        "La clave que estabas buscando",
        # Andalusian variations
        "Mira lo que he descubierto",
        "Te voy a contar una cosa",
        "Esto no lo sabe ni dios",
    ],

    # =========================================================================
    # TRANSFORMATION HOOKS - Antes/después impactante
    # =========================================================================
    "transformation": [
        "Antes y después increíble",
        "Mira la transformación",
        "De esto a esto en solo",
        "El cambio que no esperabas",
        "Transformación brutal",
        "Así empezó y así terminó",
        "El resultado te sorprenderá",
        "De cero a esto",
        "La evolución completa",
        "Cómo cambió todo",
        "El antes y después que necesitabas ver",
        "Transformación total",
        "De normal a extraordinario",
        "El proceso completo",
        "Así quedó finalmente",
    ],

    # =========================================================================
    # NUMBER HOOKS - Listas y cuantificación (alto engagement)
    # =========================================================================
    "number": [
        "3 errores que estás cometiendo",
        "5 trucos que no conocías",
        "7 cosas que deberías saber",
        "10 tips que cambiarán tu vida",
        "Los 3 secretos para",
        "5 señales de que estás haciendo bien",
        "3 razones por las que no funciona",
        "El único truco que necesitas",
        "Los 5 mejores de la ciudad",
        "3 formas de mejorar esto",
        "7 ideas que tienes que probar",
        "Los 10 imprescindibles",
        "5 pasos para conseguirlo",
        "3 claves fundamentales",
        "El top 5 definitivo",
        "Las 3 reglas de oro",
    ],

    # =========================================================================
    # BOLD CLAIM HOOKS - Afirmaciones provocativas
    # =========================================================================
    "bold_claim": [
        "Esto es lo mejor que vas a ver hoy",
        "El mejor truco que existe",
        "Nadie hace esto como nosotros",
        "Lo nunca visto",
        "Imposible no guardar esto",
        "El más increíble que he visto",
        "Esto solo pasa una vez",
        "Lo mejor de lo mejor",
        "Nunca verás algo igual",
        "El único lugar donde encontrarás esto",
        "Lo más impresionante del año",
        "Esto supera todo lo anterior",
        "El secreto definitivo",
        "La mejor inversión que harás",
        "Esto no lo encontrarás en otro sitio",
    ],

    # =========================================================================
    # STORY HOOKS - Narrativa personal (genera conexión emocional)
    # =========================================================================
    "story": [
        "Storytime: Cómo descubrí esto",
        "La historia detrás de esto",
        "Cuando empecé no tenía nada",
        "Mi experiencia con esto",
        "Un día decidí cambiar",
        "Me pasó algo increíble",
        "Nunca olvidaré el día que",
        "Todo empezó cuando",
        "Mi historia de éxito",
        "Así fue como lo conseguí",
        "La lección más importante que aprendí",
        "Mi mayor error fue",
        "Lo que me enseñó la experiencia",
        "Déjame contarte qué pasó",
        # Andalusian storytelling
        "Te cuento lo que me pasó",
        "Mira lo que me ocurrió",
        "No te vas a creer lo que pasó",
    ],

    # =========================================================================
    # HOW-TO HOOKS - Valor educativo
    # =========================================================================
    "how_to": [
        "Cómo conseguir esto en 3 pasos",
        "Aprende a hacer esto fácil",
        "Tutorial paso a paso",
        "El método más fácil para",
        "Cómo lo hago yo",
        "Así se hace correctamente",
        "La forma más sencilla de",
        "Aprende conmigo a",
        "Tutorial completo",
        "Cómo mejorar esto hoy",
        "El proceso que sigo",
        "Cómo empezar desde cero",
        "Guía definitiva para",
        "Así puedes hacerlo tú",
        "Cómo conseguir resultados",
    ],

    # =========================================================================
    # URGENCY HOOKS - FOMO y escasez
    # =========================================================================
    "urgency": [
        "Solo por tiempo limitado",
        "Últimas unidades disponibles",
        "Aprovecha ahora antes de que se agote",
        "Oferta que no se repetirá",
        "Solo hoy disponible",
        "No te lo pierdas",
        "Última oportunidad",
        "Se acaba el plazo",
        "Plazas limitadas",
        "Corre antes de que se agote",
        "Exclusivo por pocas horas",
        "No dejes pasar esta oportunidad",
        "Ahora o nunca",
        "Date prisa",
    ],

    # =========================================================================
    # SOCIAL PROOF HOOKS - Validación social
    # =========================================================================
    "social_proof": [
        "Miles de personas ya lo usan",
        "El favorito de nuestros clientes",
        "Lo que todo el mundo está haciendo",
        "El más vendido del año",
        "La tendencia que arrasa",
        "Lo que está de moda ahora",
        "El secreto de los profesionales",
        "Lo que usan los expertos",
        "Recomendado por miles",
        "El preferido del sector",
        "Lo que funciona de verdad",
        "Aprobado por los mejores",
    ],

    # =========================================================================
    # LOCAL/ANDALUSIAN SPECIFIC HOOKS
    # =========================================================================
    "local_andalusian": [
        "Esto solo lo encuentras en Sevilla",
        "Lo mejor de Andalucía",
        "El rincón más bonito de la ciudad",
        "Tradición con sabor andaluz",
        "El auténtico sabor de aquí",
        "Lo que nos hace únicos",
        "Orgullo de nuestra tierra",
        "El encanto del sur",
        "Típico de por aquí",
        "Con ese sabor que nos caracteriza",
        "Como en casa de la abuela",
        "Hecho con cariño del bueno",
    ],

    # =========================================================================
    # ENGAGEMENT DRIVER HOOKS - Específicos para interacción
    # =========================================================================
    "engagement_driver": [
        "Guarda esto para después",
        "Comparte con quien lo necesite",
        "Etiqueta a esa persona",
        "Comenta tu favorito",
        "Dime cuál prefieres",
        "¿Cuál elegirías tú?",
        "Guárdalo antes de que desaparezca",
        "Envíaselo a quien lo necesita",
        "Comenta si te identificas",
        "Dale like si estás de acuerdo",
        "Sígueme para más contenido así",
        "Activa la campanita",
    ],
}


# =============================================================================
# Flatten hooks into a single list for embedding generation
# =============================================================================

def get_all_hooks_flat() -> List[str]:
    """Get all hooks as a flat list for embedding generation."""
    all_hooks = []
    for category_hooks in VIRAL_HOOKS_DATABASE.values():
        all_hooks.extend(category_hooks)
    return all_hooks


def get_hooks_by_category() -> Dict[str, List[str]]:
    """Get hooks organized by category."""
    return VIRAL_HOOKS_DATABASE.copy()


# =============================================================================
# Semantic Hook Scorer Class
# =============================================================================

class SemanticHookScorer:
    """
    Compute semantic hook scores using embedding similarity.

    Uses SentenceTransformer (all-MiniLM-L6-v2) to generate embeddings
    for both the base hooks and input text, then computes cosine similarity.

    Features:
    - Lazy loading of embedding model
    - Caching of base hook embeddings
    - Bonus scoring for curiosity indicators
    - Category-level similarity breakdown
    """

    # Class-level singleton for embedding model
    _model = None
    _model_loaded = False

    # Cached hook embeddings
    _hook_embeddings = None
    _hook_embeddings_loaded = False
    _hooks_list = None

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize the semantic hook scorer.

        Args:
            cache_path: Path for caching hook embeddings (optional)
        """
        self.cache_path = cache_path or HOOK_EMBEDDINGS_CACHE_PATH
        self._available = self._check_availability()

        # Auto-load cached embeddings if available
        if self._available:
            self._load_cached_embeddings()

    def _check_availability(self) -> bool:
        """Check if sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Semantic hook scoring unavailable. "
                "Install with: pip install sentence-transformers"
            )
            return False

    def _load_model(self):
        """Lazy load the embedding model."""
        if SemanticHookScorer._model is not None:
            return

        if not self._available:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model for semantic hook detection...")
            SemanticHookScorer._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            SemanticHookScorer._model_loaded = True
            logger.info("Semantic hook model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model for hooks: {e}")
            SemanticHookScorer._model_loaded = False

    def _load_cached_embeddings(self):
        """Load cached hook embeddings from disk."""
        if SemanticHookScorer._hook_embeddings is not None:
            return

        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                SemanticHookScorer._hook_embeddings = cache["embeddings"]
                SemanticHookScorer._hooks_list = cache["hooks"]
                SemanticHookScorer._hook_embeddings_loaded = True
                logger.info(f"Loaded {len(SemanticHookScorer._hooks_list)} cached hook embeddings")
            except Exception as e:
                logger.warning(f"Could not load cached hook embeddings: {e}")

    def _generate_hook_embeddings(self):
        """Generate and cache embeddings for all base hooks."""
        if SemanticHookScorer._hook_embeddings is not None:
            return

        self._load_model()

        if not SemanticHookScorer._model_loaded:
            logger.warning("Cannot generate hook embeddings: model not loaded")
            return

        logger.info("Generating embeddings for base hooks corpus...")

        # Get all hooks
        hooks = get_all_hooks_flat()
        SemanticHookScorer._hooks_list = hooks

        # Generate embeddings
        embeddings = SemanticHookScorer._model.encode(
            hooks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )

        SemanticHookScorer._hook_embeddings = embeddings.astype(np.float32)
        SemanticHookScorer._hook_embeddings_loaded = True

        # Cache to disk
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache = {
                "embeddings": SemanticHookScorer._hook_embeddings,
                "hooks": SemanticHookScorer._hooks_list,
                "version": "1.0.0"
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache, f)
            logger.info(f"Cached {len(hooks)} hook embeddings to: {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not cache hook embeddings: {e}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text."""
        if not self._available or not text or not text.strip():
            return np.zeros(384, dtype=np.float32)

        self._load_model()

        if SemanticHookScorer._model is None:
            return np.zeros(384, dtype=np.float32)

        try:
            embedding = SemanticHookScorer._model.encode(
                text.strip()[:500],  # Limit text length
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return np.zeros(384, dtype=np.float32)

    def _compute_bonuses(self, text: str) -> float:
        """Compute bonus scores for curiosity indicators."""
        bonus = 0.0

        # Question bonus
        if '?' in text or '¿' in text:
            bonus += QUESTION_BONUS

        # Curiosity emoji bonus
        if any(emoji in text for emoji in CURIOSITY_EMOJIS):
            bonus += CURIOSITY_EMOJI_BONUS

        # Ellipsis/suspense bonus
        if '...' in text or '…' in text:
            bonus += ELLIPSIS_BONUS

        return bonus

    def compute_hook_score(
        self,
        caption: str,
        transcript: Optional[str] = None,
        return_details: bool = False
    ) -> Union[float, Dict]:
        """
        Compute semantic hook score for content.

        Generates embedding for caption (+ optional transcript) and computes
        max cosine similarity against all pre-defined viral hooks.

        Args:
            caption: Post caption text
            transcript: Optional video transcript (from Whisper)
            return_details: If True, return detailed breakdown

        Returns:
            If return_details=False: float score (0-1)
            If return_details=True: Dict with score, top_matches, category_scores
        """
        if not self._available:
            if return_details:
                return {
                    "score": 0.0,
                    "method": "unavailable",
                    "top_matches": [],
                    "bonuses": 0.0
                }
            return 0.0

        # Ensure hook embeddings are ready
        self._generate_hook_embeddings()

        if SemanticHookScorer._hook_embeddings is None:
            if return_details:
                return {
                    "score": 0.0,
                    "method": "fallback",
                    "top_matches": [],
                    "bonuses": 0.0
                }
            return 0.0

        # Combine caption and transcript
        if transcript and transcript.strip():
            combined_text = f"{caption.strip()} {transcript.strip()}"
        else:
            combined_text = caption.strip() if caption else ""

        if not combined_text:
            if return_details:
                return {
                    "score": 0.0,
                    "method": "empty_input",
                    "top_matches": [],
                    "bonuses": 0.0
                }
            return 0.0

        # Get embedding for input text (use first line for hook detection)
        first_line = combined_text.split('\n')[0]
        text_embedding = self._get_text_embedding(first_line)

        if np.allclose(text_embedding, 0):
            if return_details:
                return {
                    "score": 0.0,
                    "method": "embedding_failed",
                    "top_matches": [],
                    "bonuses": 0.0
                }
            return 0.0

        # Compute cosine similarities (embeddings are already normalized)
        similarities = np.dot(SemanticHookScorer._hook_embeddings, text_embedding)

        # Get max similarity
        max_idx = np.argmax(similarities)
        max_similarity = float(similarities[max_idx])

        # Compute bonuses
        bonuses = self._compute_bonuses(combined_text)

        # Final score (capped at 1.0)
        final_score = min(1.0, max_similarity + bonuses)

        if return_details:
            # Get top 3 matching hooks
            top_indices = np.argsort(similarities)[::-1][:3]
            top_matches = [
                {
                    "hook": SemanticHookScorer._hooks_list[i],
                    "similarity": float(similarities[i])
                }
                for i in top_indices
            ]

            # Compute weighted average of top-3 (alternative scoring)
            top3_avg = float(np.mean(similarities[top_indices]))

            return {
                "score": round(final_score, 4),
                "max_similarity": round(max_similarity, 4),
                "top3_avg": round(top3_avg, 4),
                "bonuses": round(bonuses, 4),
                "top_matches": top_matches,
                "method": "semantic",
                "hook_strength": self._classify_hook_strength(max_similarity)
            }

        return round(final_score, 4)

    def _classify_hook_strength(self, similarity: float) -> str:
        """Classify hook strength based on similarity score."""
        if similarity >= HIGH_HOOK_THRESHOLD:
            return "strong"
        elif similarity >= MEDIUM_HOOK_THRESHOLD:
            return "moderate"
        elif similarity >= LOW_HOOK_THRESHOLD:
            return "weak"
        else:
            return "minimal"

    def get_category_scores(self, text: str) -> Dict[str, float]:
        """
        Get hook similarity scores broken down by category.

        Useful for understanding which type of hook the content resembles.

        Args:
            text: Input text (caption or combined text)

        Returns:
            Dict mapping category name to max similarity for that category
        """
        if not self._available:
            return {}

        self._generate_hook_embeddings()

        if SemanticHookScorer._hook_embeddings is None:
            return {}

        # Get text embedding
        text_embedding = self._get_text_embedding(text.split('\n')[0])
        if np.allclose(text_embedding, 0):
            return {}

        # Compute similarities
        similarities = np.dot(SemanticHookScorer._hook_embeddings, text_embedding)

        # Map back to categories
        category_scores = {}
        idx = 0
        for category, hooks in VIRAL_HOOKS_DATABASE.items():
            category_sims = similarities[idx:idx + len(hooks)]
            category_scores[category] = float(np.max(category_sims))
            idx += len(hooks)

        return category_scores

    @property
    def is_available(self) -> bool:
        """Check if semantic hook scoring is available."""
        return self._available

    @property
    def is_ready(self) -> bool:
        """Check if scorer is fully initialized with embeddings."""
        return self._available and SemanticHookScorer._hook_embeddings_loaded


# =============================================================================
# Global Instance (Singleton)
# =============================================================================

_semantic_hook_scorer: Optional[SemanticHookScorer] = None


def get_semantic_hook_scorer() -> SemanticHookScorer:
    """Get global semantic hook scorer instance."""
    global _semantic_hook_scorer
    if _semantic_hook_scorer is None:
        _semantic_hook_scorer = SemanticHookScorer()
    return _semantic_hook_scorer


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_hook_score(
    caption: str,
    transcript: Optional[str] = None,
    return_details: bool = False
) -> Union[float, Dict]:
    """
    Compute semantic hook score for content.

    This is the main entry point for hook scoring.

    Args:
        caption: Post caption text
        transcript: Optional video transcript
        return_details: Return detailed breakdown

    Returns:
        Hook score (0-1) or detailed dict if return_details=True

    Example:
        >>> score = compute_hook_score("POV: Descubres el mejor café de la ciudad")
        >>> print(score)  # ~0.85+

        >>> details = compute_hook_score("Algo random sin hook", return_details=True)
        >>> print(details["hook_strength"])  # "minimal"
    """
    return get_semantic_hook_scorer().compute_hook_score(
        caption, transcript, return_details
    )


def get_hook_category_scores(text: str) -> Dict[str, float]:
    """
    Get hook similarity scores by category.

    Args:
        text: Input text

    Returns:
        Dict mapping category to max similarity score
    """
    return get_semantic_hook_scorer().get_category_scores(text)


# =============================================================================
# RegEx Fallback (Kept as auxiliary feature)
# =============================================================================

REGEX_HOOK_PATTERNS = {
    "pov": re.compile(r"pov[:\s]|punto de vista", re.IGNORECASE),
    "question": re.compile(r"^[¿?]|^\w+\s*\?", re.IGNORECASE),
    "number": re.compile(r"^\d+\s+\w+|top\s*\d+|\d+\s*(cosas|tips|errores|razones)", re.IGNORECASE),
    "bold_claim": re.compile(r"nunca|siempre|todo|nadie|el mejor|el peor|imposible", re.IGNORECASE),
    "story": re.compile(r"historia|storytime|cuando|un día|me pasó", re.IGNORECASE),
    "how_to": re.compile(r"cómo\s+\w+|aprende\s+a|tutorial|paso\s+a\s+paso", re.IGNORECASE),
    "reveal": re.compile(r"secreto|te cuento|descubre|te revelo|no sabías", re.IGNORECASE),
}


def compute_regex_hook_score(caption: str) -> Tuple[float, List[str]]:
    """
    Compute hook score using RegEx patterns (fallback/auxiliary).

    Args:
        caption: Post caption text

    Returns:
        Tuple of (score, matched_patterns)
    """
    if not caption:
        return 0.0, []

    first_line = caption.split('\n')[0].lower()
    matches = []

    for hook_type, pattern in REGEX_HOOK_PATTERNS.items():
        if pattern.search(first_line):
            matches.append(hook_type)

    # Score based on number of matches
    if len(matches) >= 3:
        score = 0.9
    elif len(matches) == 2:
        score = 0.7
    elif len(matches) == 1:
        score = 0.5
    else:
        score = 0.1

    return score, matches


# =============================================================================
# Combined Hook Scoring (Semantic + RegEx ensemble)
# =============================================================================

def compute_combined_hook_score(
    caption: str,
    transcript: Optional[str] = None,
    semantic_weight: float = 0.85,
    regex_weight: float = 0.15
) -> Dict[str, any]:
    """
    Compute combined hook score using both semantic and RegEx methods.

    Semantic scoring is primary (85% weight) with RegEx as backup (15% weight).

    Args:
        caption: Post caption
        transcript: Optional transcript
        semantic_weight: Weight for semantic score (default 0.85)
        regex_weight: Weight for regex score (default 0.15)

    Returns:
        Dict with combined_score, semantic_score, regex_score, details
    """
    # Semantic score
    semantic_details = compute_hook_score(caption, transcript, return_details=True)
    semantic_score = semantic_details.get("score", 0.0)

    # RegEx score
    regex_score, regex_matches = compute_regex_hook_score(caption)

    # Combined score (weighted average)
    combined_score = (semantic_score * semantic_weight) + (regex_score * regex_weight)

    return {
        "combined_score": round(combined_score, 4),
        "semantic_score": round(semantic_score, 4),
        "regex_score": round(regex_score, 4),
        "semantic_details": semantic_details,
        "regex_matches": regex_matches,
        "hook_strength": semantic_details.get("hook_strength", "unknown"),
        "method": "ensemble"
    }


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("SEMANTIC HOOK DETECTION - BrandPulse AI")
    print("=" * 70 + "\n")

    # Test cases: hooks creativos vs contenido neutro
    test_cases = [
        # High-hook cases (should score > 0.7)
        {
            "caption": "No vas a creer lo que descubrí hoy en este rincón de Sevilla 😱",
            "expected": "high",
            "description": "Reveal hook con ubicación local"
        },
        {
            "caption": "POV: Llegas a tu café favorito y huele a recién hecho ☕",
            "expected": "high",
            "description": "POV hook inmersivo"
        },
        {
            "caption": "El truco que nadie te cuenta para conseguir el mejor resultado",
            "expected": "high",
            "description": "Reveal/secret hook clásico"
        },
        {
            "caption": "3 errores que cometes sin darte cuenta (el último te sorprenderá)",
            "expected": "high",
            "description": "Number hook + curiosity"
        },
        {
            "caption": "Mira esta transformación brutal, de esto a esto en solo 2 semanas 🔥",
            "expected": "high",
            "description": "Transformation hook"
        },
        {
            "caption": "¿Sabías que esto existe? Lo acabo de descubrir y flipé 🤯",
            "expected": "high",
            "description": "Question hook + reveal"
        },
        # Creative variations (should still score > 0.5)
        {
            "caption": "De qué manera puedes mejorar esto en tu negocio",
            "expected": "medium",
            "description": "Creative how-to (sin 'cómo')"
        },
        {
            "caption": "Esto es lo que los profesionales no quieren que sepas",
            "expected": "high",
            "description": "Secret variation"
        },
        # Low-hook cases (should score < 0.3)
        {
            "caption": "Hoy hemos abierto a las 9:00. Os esperamos.",
            "expected": "low",
            "description": "Informativo neutro"
        },
        {
            "caption": "Nuevo producto disponible en tienda",
            "expected": "low",
            "description": "Anuncio simple"
        },
        {
            "caption": "Feliz lunes a todos",
            "expected": "low",
            "description": "Saludo genérico"
        },
    ]

    print("[1] Initializing Semantic Hook Scorer...")
    scorer = get_semantic_hook_scorer()
    print(f"    Available: {scorer.is_available}")

    if not scorer.is_available:
        print("\n[ERROR] Sentence-transformers not available. Install with:")
        print("    pip install sentence-transformers")
        sys.exit(1)

    print("\n[2] Running test cases...\n")
    print("-" * 70)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        result = compute_hook_score(test["caption"], return_details=True)
        score = result["score"]
        strength = result["hook_strength"]

        # Determine if test passed
        if test["expected"] == "high":
            success = score >= 0.7 or strength == "strong"
        elif test["expected"] == "medium":
            success = score >= 0.5
        else:  # low
            success = score < 0.4

        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
        else:
            failed += 1

        print(f"Test {i}: [{status}]")
        print(f"  Caption: {test['caption'][:60]}...")
        print(f"  Expected: {test['expected']} | Got: {strength} (score: {score:.3f})")
        print(f"  Description: {test['description']}")
        if result.get("top_matches"):
            top_match = result["top_matches"][0]
            print(f"  Top match: '{top_match['hook'][:40]}...' ({top_match['similarity']:.3f})")
        print()

    print("-" * 70)
    print(f"\n[3] Results: {passed}/{len(test_cases)} tests passed\n")

    if failed > 0:
        print(f"[WARNING] {failed} tests failed. Review hook corpus or thresholds.")

    # Detailed example
    print("\n[4] Detailed analysis example:")
    example_caption = "El secreto mejor guardado de Triana: este café te cambia la vida ☕🔥"
    details = compute_hook_score(example_caption, return_details=True)
    print(f"\n  Caption: '{example_caption}'")
    print(f"\n  Score: {details['score']}")
    print(f"  Max similarity: {details['max_similarity']}")
    print(f"  Top-3 average: {details['top3_avg']}")
    print(f"  Bonuses: {details['bonuses']}")
    print(f"  Hook strength: {details['hook_strength']}")
    print(f"\n  Top matches:")
    for match in details['top_matches']:
        print(f"    - {match['hook']} ({match['similarity']:.3f})")

    # Category breakdown
    print("\n[5] Category breakdown:")
    category_scores = get_hook_category_scores(example_caption)
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    for cat, score in sorted_cats[:5]:
        print(f"    {cat}: {score:.3f}")

    print("\n" + "=" * 70)
    print("SEMANTIC HOOK DETECTION TEST COMPLETE")
    print("=" * 70 + "\n")

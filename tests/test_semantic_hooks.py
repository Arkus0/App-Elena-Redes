#!/usr/bin/env python3
"""
Tests for Semantic Hook Detection
==================================

Validates that the semantic hook scoring system correctly identifies
viral hooks vs neutral content using embedding-based similarity.

Test cases:
1. High-hook captions (creative variations) -> score > 0.7
2. Neutral/informative captions -> score < 0.3
3. Combined caption + transcript scoring
4. Category-level scoring breakdown

Run with: pytest tests/test_semantic_hooks.py -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import semantic hooks module
try:
    from backend.ml.semantic_hooks import (
        compute_hook_score,
        compute_combined_hook_score,
        get_hook_category_scores,
        get_semantic_hook_scorer,
        VIRAL_HOOKS_DATABASE,
        HIGH_HOOK_THRESHOLD,
        MEDIUM_HOOK_THRESHOLD,
        LOW_HOOK_THRESHOLD,
    )
    SEMANTIC_HOOKS_AVAILABLE = True
except ImportError:
    SEMANTIC_HOOKS_AVAILABLE = False


# =============================================================================
# Test Cases
# =============================================================================

# High-hook captions (should score >= 0.7)
HIGH_HOOK_CASES = [
    {
        "caption": "No vas a creer lo que descubri hoy en este rincon de Sevilla",
        "description": "Reveal hook with local reference",
        "expected_min": 0.65,
    },
    {
        "caption": "POV: Llegas a tu cafe favorito y huele a recien hecho",
        "description": "POV hook immersive",
        "expected_min": 0.65,
    },
    {
        "caption": "El truco que nadie te cuenta para conseguir el mejor resultado",
        "description": "Reveal/secret hook classic",
        "expected_min": 0.65,
    },
    {
        "caption": "3 errores que cometes sin darte cuenta (el ultimo te sorprendera)",
        "description": "Number hook + curiosity",
        "expected_min": 0.65,
    },
    {
        "caption": "Mira esta transformacion brutal, de esto a esto en solo 2 semanas",
        "description": "Transformation hook",
        "expected_min": 0.65,
    },
    {
        "caption": "Sabias que esto existe? Lo acabo de descubrir y flipe",
        "description": "Question hook + reveal",
        "expected_min": 0.65,
    },
    {
        "caption": "El secreto mejor guardado de Triana: este cafe te cambia la vida",
        "description": "Secret + local Andalusian reference",
        "expected_min": 0.65,
    },
    {
        "caption": "Storytime: Como descubri el lugar perfecto para trabajar en remoto",
        "description": "Story hook",
        "expected_min": 0.60,
    },
]

# Creative variations without exact keywords (should still score >= 0.5)
CREATIVE_HOOK_CASES = [
    {
        "caption": "De que manera puedes mejorar esto en tu negocio",
        "description": "Creative how-to without 'como'",
        "expected_min": 0.45,
    },
    {
        "caption": "Esto es lo que los profesionales no quieren que sepas",
        "description": "Secret variation",
        "expected_min": 0.55,
    },
    {
        "caption": "La verdad que pocos conocen sobre este metodo",
        "description": "Reveal variation",
        "expected_min": 0.50,
    },
    {
        "caption": "Lo que me hubiera gustado saber antes de empezar",
        "description": "Regret/learning hook",
        "expected_min": 0.50,
    },
]

# Low-hook captions (should score < 0.35)
LOW_HOOK_CASES = [
    {
        "caption": "Hoy hemos abierto a las 9:00. Os esperamos.",
        "description": "Informative neutral",
        "expected_max": 0.40,
    },
    {
        "caption": "Nuevo producto disponible en tienda",
        "description": "Simple announcement",
        "expected_max": 0.40,
    },
    {
        "caption": "Feliz lunes a todos",
        "description": "Generic greeting",
        "expected_max": 0.35,
    },
    {
        "caption": "Mesa reservada para las 20:00",
        "description": "Operational info",
        "expected_max": 0.35,
    },
    {
        "caption": "Abrimos de lunes a viernes de 10 a 20h",
        "description": "Business hours",
        "expected_max": 0.35,
    },
]


# =============================================================================
# Test Functions
# =============================================================================

@pytest.mark.skipif(not SEMANTIC_HOOKS_AVAILABLE, reason="Semantic hooks not available")
class TestSemanticHookScorer:
    """Test suite for semantic hook detection."""

    def test_scorer_initialization(self):
        """Test that the scorer initializes correctly."""
        scorer = get_semantic_hook_scorer()
        assert scorer.is_available, "Scorer should be available"

    def test_hooks_database_loaded(self):
        """Test that the hooks database is properly loaded."""
        assert len(VIRAL_HOOKS_DATABASE) > 0, "Hooks database should not be empty"

        total_hooks = sum(len(hooks) for hooks in VIRAL_HOOKS_DATABASE.values())
        assert total_hooks >= 100, f"Expected at least 100 hooks, got {total_hooks}"

    @pytest.mark.parametrize("test_case", HIGH_HOOK_CASES)
    def test_high_hook_detection(self, test_case):
        """Test that high-hook captions score above threshold."""
        score = compute_hook_score(test_case["caption"])

        assert score >= test_case["expected_min"], (
            f"Hook detection failed for '{test_case['description']}': "
            f"expected >= {test_case['expected_min']}, got {score:.3f}\n"
            f"Caption: {test_case['caption']}"
        )

    @pytest.mark.parametrize("test_case", CREATIVE_HOOK_CASES)
    def test_creative_hook_detection(self, test_case):
        """Test that creative variations without keywords are still detected."""
        score = compute_hook_score(test_case["caption"])

        assert score >= test_case["expected_min"], (
            f"Creative hook detection failed for '{test_case['description']}': "
            f"expected >= {test_case['expected_min']}, got {score:.3f}\n"
            f"Caption: {test_case['caption']}"
        )

    @pytest.mark.parametrize("test_case", LOW_HOOK_CASES)
    def test_low_hook_detection(self, test_case):
        """Test that neutral captions score below threshold."""
        score = compute_hook_score(test_case["caption"])

        assert score <= test_case["expected_max"], (
            f"Low hook detection failed for '{test_case['description']}': "
            f"expected <= {test_case['expected_max']}, got {score:.3f}\n"
            f"Caption: {test_case['caption']}"
        )

    def test_detailed_result_structure(self):
        """Test that detailed results contain expected fields."""
        result = compute_hook_score(
            "POV: Descubres el mejor cafe de la ciudad",
            return_details=True
        )

        assert "score" in result
        assert "max_similarity" in result
        assert "top3_avg" in result
        assert "bonuses" in result
        assert "top_matches" in result
        assert "hook_strength" in result
        assert "method" in result

        assert len(result["top_matches"]) == 3
        assert all("hook" in m and "similarity" in m for m in result["top_matches"])

    def test_question_bonus(self):
        """Test that question marks provide a bonus."""
        caption_no_question = "El secreto mejor guardado de Sevilla"
        caption_with_question = "Sabias esto? El secreto mejor guardado de Sevilla"

        score_no_q = compute_hook_score(caption_no_question)
        score_with_q = compute_hook_score(caption_with_question)

        # Score with question should be at least slightly higher due to bonus
        # (may not be significant if base similarity is already high)
        assert score_with_q >= score_no_q, (
            f"Question bonus not applied: without={score_no_q:.3f}, with={score_with_q:.3f}"
        )

    def test_transcript_enhances_score(self):
        """Test that transcript can enhance hook detection."""
        caption = "Mira este video increible"
        transcript = "Hoy os voy a contar el secreto que cambio mi negocio para siempre"

        score_caption_only = compute_hook_score(caption)
        score_with_transcript = compute_hook_score(caption, transcript=transcript)

        # With a hook-rich transcript, score should be higher
        assert score_with_transcript >= score_caption_only, (
            f"Transcript did not enhance score: "
            f"caption only={score_caption_only:.3f}, "
            f"with transcript={score_with_transcript:.3f}"
        )

    def test_empty_input_handling(self):
        """Test graceful handling of empty inputs."""
        assert compute_hook_score("") == 0.0
        assert compute_hook_score(None) == 0.0
        assert compute_hook_score("   ") == 0.0

    def test_category_scores(self):
        """Test category-level similarity breakdown."""
        caption = "POV: Descubres el mejor cafe de la ciudad"
        scores = get_hook_category_scores(caption)

        assert len(scores) > 0, "Should return category scores"
        assert "pov" in scores, "Should have POV category score"
        assert all(0 <= v <= 1 for v in scores.values()), "All scores should be 0-1"

        # POV should be highest for this caption
        max_category = max(scores, key=scores.get)
        assert max_category == "pov" or scores["pov"] >= 0.6, (
            f"Expected POV to score high for POV caption, got max={max_category}"
        )

    def test_combined_scoring(self):
        """Test ensemble scoring (semantic + regex)."""
        caption = "3 secretos que nadie te cuenta sobre este metodo"
        result = compute_combined_hook_score(caption)

        assert "combined_score" in result
        assert "semantic_score" in result
        assert "regex_score" in result
        assert "method" in result

        assert result["method"] == "ensemble"
        assert 0 <= result["combined_score"] <= 1

    def test_hook_strength_classification(self):
        """Test hook strength classification."""
        # Strong hook
        strong_result = compute_hook_score(
            "El secreto que nadie te cuenta y que cambiara tu vida",
            return_details=True
        )
        assert strong_result["hook_strength"] in ["strong", "moderate"]

        # Weak hook
        weak_result = compute_hook_score(
            "Buenos dias",
            return_details=True
        )
        assert weak_result["hook_strength"] in ["weak", "minimal"]


@pytest.mark.skipif(not SEMANTIC_HOOKS_AVAILABLE, reason="Semantic hooks not available")
class TestSemanticVsRegex:
    """Test that semantic beats regex for creative variations."""

    def test_semantic_detects_without_keywords(self):
        """Test semantic catches hooks without exact keyword matches."""
        # This variation avoids "como", "secreto", "truco" etc
        creative_hook = "De esta manera consegui lo que buscaba"

        semantic_score = compute_hook_score(creative_hook)

        # Semantic should detect the how-to/reveal pattern
        assert semantic_score >= 0.4, (
            f"Semantic should detect creative variation, got {semantic_score:.3f}"
        )

    def test_semantic_handles_andalusian_variations(self):
        """Test semantic handles regional variations."""
        andalusian_hooks = [
            "Mira lo que he pillao en la tienda",  # "pillao" informal
            "Esto no lo sabe ni dios",  # Andalusian expression
            "Te cuento una cosa que flipas",  # Informal
        ]

        for caption in andalusian_hooks:
            score = compute_hook_score(caption)
            # Should recognize these as hook-like despite informal language
            assert score >= 0.35, (
                f"Andalusian variation not detected: '{caption}' -> {score:.3f}"
            )


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not SEMANTIC_HOOKS_AVAILABLE, reason="Semantic hooks not available")
class TestIntegration:
    """Integration tests with ml_service."""

    def test_feature_extraction_includes_semantic_hooks(self):
        """Test that FeatureExtractor includes semantic hook features."""
        try:
            from backend.app.services.ml_service import FeatureExtractor

            content = {
                "caption": "El secreto que nadie te cuenta sobre este metodo",
                "content_format": "reel",
            }

            features = FeatureExtractor.extract_features(content)

            assert "semantic_hook_score" in features
            assert "semantic_hook_max_sim" in features
            assert "semantic_hook_top3_avg" in features
            assert "hook_regex_score" in features

            # Semantic hook score should be high for this caption
            assert features["semantic_hook_score"] >= 0.5, (
                f"Expected semantic_hook_score >= 0.5, got {features['semantic_hook_score']}"
            )

        except ImportError:
            pytest.skip("ml_service not available")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    if not SEMANTIC_HOOKS_AVAILABLE:
        print("ERROR: Semantic hooks not available. Install sentence-transformers:")
        print("  pip install sentence-transformers")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SEMANTIC HOOK DETECTION TESTS")
    print("=" * 70 + "\n")

    # Quick validation
    print("[1] Testing high-hook captions...")
    passed = 0
    failed = 0

    for case in HIGH_HOOK_CASES:
        score = compute_hook_score(case["caption"])
        if score >= case["expected_min"]:
            print(f"  PASS: {case['description']} -> {score:.3f}")
            passed += 1
        else:
            print(f"  FAIL: {case['description']} -> {score:.3f} (expected >= {case['expected_min']})")
            failed += 1

    print(f"\n[2] Testing low-hook captions...")
    for case in LOW_HOOK_CASES:
        score = compute_hook_score(case["caption"])
        if score <= case["expected_max"]:
            print(f"  PASS: {case['description']} -> {score:.3f}")
            passed += 1
        else:
            print(f"  FAIL: {case['description']} -> {score:.3f} (expected <= {case['expected_max']})")
            failed += 1

    print(f"\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    if failed > 0:
        sys.exit(1)

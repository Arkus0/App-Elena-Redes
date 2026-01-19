import sys
import os
import time
import functools
import psutil
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
# Ensure project root is FIRST to resolve 'ml' to /app/ml, not /app/backend/ml
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Add backend to path for 'app' imports, but append it so it has lower priority than root
# actually, if we are at root, we can import 'backend.app' if 'backend' is a package?
# But 'app' is inside 'backend'. Typical pattern is to run from backend/ or add backend/ to path.
# We add it at the end (or index 1)
backend_path = str(PROJECT_ROOT / "backend")
if backend_path not in sys.path:
    sys.path.insert(1, backend_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def stress_test():
    logger.info("Starting stress test...")
    logger.info(f"sys.path: {sys.path[:3]}")

    baseline_mem = get_process_memory()
    logger.info(f"Baseline Memory: {baseline_mem:.2f} MB")

    # Import MLPredictor (should be lazy now)
    try:
        from app.services.ml_service import MLPredictor
    except ImportError:
        # Fallback if app not found directly, try backend.app
        try:
            from backend.app.services.ml_service import MLPredictor
        except ImportError:
             logger.error("Could not import MLPredictor. Check paths.")
             sys.exit(1)

    logger.info("Instantiating MLPredictor (should be cheap)...")
    predictor = MLPredictor()

    after_init_mem = get_process_memory()
    logger.info(f"After Init Memory: {after_init_mem:.2f} MB")

    # We allow some overhead for library imports (torch, pandas) which happen at module level
    # But we want to ensure no MODEL WEIGHTS are loaded.
    # Ideally, we would measure before import, but imports are global.

    # Simulate prediction to trigger loading
    content = {
        "caption": "Test content for prediction " * 5,
        "business_type": "cafe",
        "likes_count": 100,
        "comments_count": 10
    }

    logger.info("Triggering first prediction (loading models)...")
    start_time = time.time()
    result = predictor.predict_engagement(content)
    duration = time.time() - start_time

    after_load_mem = get_process_memory()
    logger.info(f"After Prediction Memory: {after_load_mem:.2f} MB")
    logger.info(f"First prediction took: {duration:.4f}s")

    # Test LRU Cache on EmbeddingExtractor
    try:
        from ml.features_embeddings import get_embedding_extractor
    except ImportError:
        logger.error("Could not import features_embeddings from ml package.")
        sys.exit(1)

    extractor = get_embedding_extractor("low")
    test_text = "Caching verification text"

    logger.info("Testing LRU Cache...")

    # First call (uncached)
    start_time = time.time()
    emb1 = extractor.get_raw_embedding(test_text)
    duration1 = time.time() - start_time
    logger.info(f"1st Embedding Call: {duration1:.6f}s")

    # Second call (cached)
    start_time = time.time()
    emb2 = extractor.get_raw_embedding(test_text)
    duration2 = time.time() - start_time
    logger.info(f"2nd Embedding Call: {duration2:.6f}s")

    if duration2 < duration1 * 0.1 or duration2 < 0.0001:
        logger.info("✅ LRU Cache working: 2nd call significantly faster.")
    elif duration1 < 0.001:
        logger.info("⚠️ First call was too fast to compare (mocked or no model?), assuming cache works if logic is correct.")
    else:
        logger.warning(f"❌ LRU Cache might not be working efficiently. Speedup: {duration1/duration2:.2f}x")

    logger.info("Stress test complete.")

if __name__ == "__main__":
    stress_test()

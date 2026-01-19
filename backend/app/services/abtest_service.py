
import random
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.abtest import ABTestVariant

def perform_thompson_sampling(experiment) -> str:
    """
    Selects the best variant for an experiment using Thompson Sampling.
    Returns the 'variant_name' of the winner.

    Args:
        experiment: The ABTestExperiment object with variants already loaded.
                    (We pass the object to avoid async DB calls in synchronous contexts
                     or redundant queries if already loaded via selectinload).
    """
    if not experiment or not experiment.variants:
        return "reel"  # Fallback

    # Sample from Beta Distribution for each variant
    best_variant = None
    max_sample = -1.0

    for variant in experiment.variants:
        # random.betavariate(alpha, beta)
        # Ensure alpha/beta > 0
        alpha = max(1, variant.alpha_param)
        beta = max(1, variant.beta_param)

        sample = random.betavariate(alpha, beta)

        if sample > max_sample:
            max_sample = sample
            best_variant = variant

    return best_variant.variant_name if best_variant else "reel"

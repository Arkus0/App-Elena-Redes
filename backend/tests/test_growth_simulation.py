from app.services.growth_prediction_engine import GrowthPredictionEngine
from datetime import datetime, timedelta

def test_simulate_growth():
    engine = GrowthPredictionEngine()

    current_followers = 1000
    base_rate = 0.001 # 0.1% daily

    # 3 scheduled posts
    scheduled = [
        {"date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"), "rpi": 1.0},
        {"date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"), "rpi": 1.6}, # Viral
        {"date": (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"), "rpi": 1.2}, # High Perf
    ]

    result = engine.simulate_growth(current_followers, base_rate, scheduled)

    assert result['current_followers'] == 1000
    assert len(result['projection']) == 30
    assert result['projected_total_gain'] > 0

    # Check viral spike
    viral_day = result['projection'][4] # Day 5 (index 4 because loop starts i=0 -> day=1) - wait, target_date = today + i + 1.
    # i=0 -> today+1. i=1 -> today+2. i=4 -> today+5.

    day_5_proj = next(p for p in result['projection'] if p['date'] == scheduled[1]['date'])
    assert day_5_proj['scenario'] == 'viral_spike'

    print("Simulation test passed!")
    print(f"Total Gain: {result['projected_total_gain']}")

if __name__ == "__main__":
    test_simulate_growth()

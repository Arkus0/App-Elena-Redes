
import sys
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# Add ml folder to sys.path to import train module directly
sys.path.append(str(PROJECT_ROOT / "ml"))

try:
    # Try importing as module first (if ml/ is in path)
    from train import load_data_from_database
except ImportError:
    # Fallback if it's treated as package
    from ml.train import load_data_from_database

@patch('train.create_engine')
@patch('train.pd.read_sql')
@patch('train.DB_URL', "sqlite:///test.db")
def test_load_data_from_database(mock_read_sql, mock_create_engine):
    # Mock DataFrame returned by SQL query
    mock_df = pd.DataFrame({
        'caption': ['Test caption', None],
        'likes': [100, 50],
        'comments': [10, 5],
        'shares': [5, 2],
        'saves': [2, 1],
        'engagement_rate': [5.5, 2.3],
        'content_format': ['reel', 'static_image'],
        'is_viral': [True, False],
        'video_duration': [30, None],
        'business_type': ['restaurante', 'restaurante']
    })
    mock_read_sql.return_value = mock_df

    # Call function
    df = load_data_from_database(niche='restaurante')

    # Verify engine creation
    mock_create_engine.assert_called_once_with("sqlite:///test.db")

    # Verify SQL query execution
    args, kwargs = mock_read_sql.call_args
    query = args[0]
    assert "SELECT" in str(query)
    assert "FROM scraped_posts sp" in str(query)
    assert "JOIN competitors c" in str(query)
    assert "JOIN businesses b" in str(query)
    assert "b.business_type = :niche" in str(query)
    assert kwargs['params'] == {'niche': 'restaurante'}

    # Verify post-processing
    assert 'is_reel' in df.columns
    assert 'is_static' in df.columns

    # Check is_reel logic
    assert df.iloc[0]['is_reel'] == 1 # 'reel'
    assert df.iloc[1]['is_reel'] == 0 # 'static_image'

    # Check is_static logic
    assert df.iloc[0]['is_static'] == 0
    assert df.iloc[1]['is_static'] == 1

    # Check caption fillna
    assert df.iloc[1]['caption'] == ""

def test_load_data_from_database_no_url():
    with patch('train.DB_URL', None):
        df = load_data_from_database(niche='restaurante')
        assert df.empty

"""
Replay script: Re-process raw data through feature pipeline to verify reproducibility.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import features module
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.featurizer import FeatureComputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(raw_paths: list) -> list:
    """
    Load raw tick data from NDJSON files.
    
    Args:
        raw_paths: List of file paths or glob patterns
        
    Returns:
        List of tick dictionaries
    """
    ticks = []
    
    for pattern in raw_paths:
        # Handle glob patterns
        if '*' in pattern:
            files = Path('.').glob(pattern)
        else:
            files = [Path(pattern)]
        
        for filepath in files:
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            logger.info(f"Loading {filepath}")
            
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        tick = json.loads(line.strip())
                        ticks.append(tick)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue
    
    logger.info(f"Loaded {len(ticks)} total ticks")
    return ticks


def replay_features(ticks: list, window_sizes: list = [30, 60, 300]) -> pd.DataFrame:
    """
    Replay ticks through feature computation.
    
    Args:
        ticks: List of tick dictionaries
        window_sizes: Window sizes in seconds
        
    Returns:
        DataFrame of computed features
    """
    feature_computer = FeatureComputer(window_sizes=window_sizes)
    features_list = []
    
    logger.info("Computing features...")
    
    for i, tick in enumerate(ticks):
        # Add tick to buffer
        feature_computer.add_tick(tick)
        
        # Compute features
        features = feature_computer.compute_features(tick)
        features_list.append(features)
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(ticks)} ticks")
    
    logger.info(f"Computed features for {len(features_list)} ticks")
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    return df


def compare_outputs(replay_df: pd.DataFrame, original_file: str):
    """
    Compare replay output with original live features.
    
    Args:
        replay_df: Features from replay
        original_file: Path to original features parquet
    """
    if not Path(original_file).exists():
        logger.warning(f"Original file not found: {original_file}")
        logger.info("Cannot compare outputs.")
        return
    
    logger.info(f"Loading original features from {original_file}")
    original_df = pd.read_parquet(original_file)
    
    # Basic comparison
    logger.info(f"\nComparison:")
    logger.info(f"  Replay rows: {len(replay_df)}")
    logger.info(f"  Original rows: {len(original_df)}")
    
    # Check column names
    replay_cols = set(replay_df.columns)
    original_cols = set(original_df.columns)
    
    if replay_cols == original_cols:
        logger.info(f"  ✓ Column names match ({len(replay_cols)} columns)")
    else:
        logger.warning(f"  ✗ Column mismatch!")
        logger.warning(f"    Only in replay: {replay_cols - original_cols}")
        logger.warning(f"    Only in original: {original_cols - replay_cols}")
    
    # Compare numeric columns (within tolerance due to floating point)
    if len(replay_df) == len(original_df):
        numeric_cols = replay_df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            if col in original_df.columns:
                # Drop NaN for comparison
                replay_vals = replay_df[col].dropna()
                original_vals = original_df[col].dropna()
                
                if len(replay_vals) > 0 and len(original_vals) > 0:
                    # Check if values are close (within 1e-6)
                    if len(replay_vals) == len(original_vals):
                        close = pd.Series(replay_vals.values).sub(original_vals.values).abs() < 1e-6
                        match_pct = close.sum() / len(close) * 100
                        
                        if match_pct >= 99.9:
                            logger.info(f"  ✓ {col}: {match_pct:.2f}% match")
                        else:
                            logger.warning(f"  ⚠ {col}: {match_pct:.2f}% match")
    else:
        logger.warning("  Cannot compare values: different number of rows")


def main():
    parser = argparse.ArgumentParser(description='Replay raw data through feature pipeline')
    parser.add_argument('--raw', nargs='+', required=True,
                        help='Path(s) to raw NDJSON files (supports glob patterns)')
    parser.add_argument('--out', default='data/processed/features_replay.parquet',
                        help='Output parquet file for replayed features')
    parser.add_argument('--windows', nargs='+', type=int, default=[30, 60, 300],
                        help='Window sizes in seconds')
    parser.add_argument('--compare', default='data/processed/features.parquet',
                        help='Original features file to compare against')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    ticks = load_raw_data(args.raw)
    
    if not ticks:
        logger.error("No ticks loaded. Exiting.")
        return
    
    # Replay features
    features_df = replay_features(ticks, window_sizes=args.windows)
    
    # Save replayed features
    logger.info(f"Saving replayed features to {args.out}")
    features_df.to_parquet(args.out, index=False)
    logger.info(f"✓ Saved {len(features_df)} rows")
    
    # Compare with original if available
    if args.compare:
        compare_outputs(features_df, args.compare)


if __name__ == '__main__':
    main()
"""
Generate Evidently report to monitor data quality and drift.
Compares early vs late windows of collected data.
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import timedelta

from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset
from evidently.metrics import *

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_split_data(features_path: str, split_ratio: float = 0.5):
    """
    Load features and split into reference (early) and current (late) datasets.
    
    Args:
        features_path: Path to features parquet file
        split_ratio: Fraction of data to use as reference (default 0.5)
        
    Returns:
        reference_df, current_df
    """
    logger.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    # Convert timestamp if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Split into reference (early) and current (late)
    split_idx = int(len(df) * split_ratio)
    
    reference = df.iloc[:split_idx].copy()
    current = df.iloc[split_idx:].copy()
    
    logger.info(f"Split data:")
    logger.info(f"  Reference: {len(reference)} rows")
    logger.info(f"  Current: {len(current)} rows")
    
    if 'timestamp' in df.columns:
        logger.info(f"  Reference time: {reference['timestamp'].min()} to {reference['timestamp'].max()}")
        logger.info(f"  Current time: {current['timestamp'].min()} to {current['timestamp'].max()}")
    
    return reference, current


def select_features_for_drift(df: pd.DataFrame) -> list:
    """
    Select numeric features for drift analysis.
    
    Args:
        df: DataFrame with features
        
    Returns:
        List of column names to analyze
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclude certain columns
    exclude = ['timestamp', 'label', 'future_volatility']
    feature_cols = [col for col in numeric_cols if col not in exclude]
    
    logger.info(f"Selected {len(feature_cols)} features for drift analysis")
    return feature_cols


def generate_report(reference_df: pd.DataFrame, 
                   current_df: pd.DataFrame,
                   output_path: str = 'reports/evidently/data_drift_report.html'):
    """
    Generate Evidently report comparing reference and current data.
    
    Args:
        reference_df: Reference (early) dataset
        current_df: Current (late) dataset
        output_path: Where to save the HTML report
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating Evidently report...")
    
    # Create report with data quality and drift analysis
    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
    ])
    
    # Run report
    report.run(reference_data=reference_df, current_data=current_df)
    
    # Save as HTML
    report.save_html(output_path)
    logger.info(f"✓ Report saved to {output_path}")
    
    # Also save as JSON for programmatic access
    json_path = output_path.replace('.html', '.json')
    report.save_json(json_path)
    logger.info(f"✓ JSON report saved to {json_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Generate Evidently data drift report')
    parser.add_argument('--features', 
                       default='data/processed/features.parquet',
                       help='Path to features parquet file')
    parser.add_argument('--output',
                       default='reports/evidently/data_drift_report.html',
                       help='Output path for HTML report')
    parser.add_argument('--split_ratio',
                       type=float,
                       default=0.5,
                       help='Fraction of data to use as reference (default 0.5)')
    
    args = parser.parse_args()
    
    # Load and split data
    reference, current = load_and_split_data(args.features, args.split_ratio)
    
    # Generate report
    report = generate_report(reference, current, args.output)
    
    logger.info("Done! Open the HTML report in your browser to view results.")


if __name__ == '__main__':
    main()
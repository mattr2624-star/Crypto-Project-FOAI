"""
Milestone 2 Verification Script
Checks that all required deliverables are present and valid.
"""

from pathlib import Path
import pandas as pd
import json

def check_file(path: str, description: str) -> bool:
    """Check if a file exists."""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists

def check_parquet_file(path: str, min_rows: int = 100) -> bool:
    """Check if parquet file exists and has minimum rows."""
    if not Path(path).exists():
        print(f"  ✗ Parquet file not found: {path}")
        return False
    
    try:
        df = pd.read_parquet(path)
        if len(df) >= min_rows:
            print(f"  ✓ {path}: {len(df)} rows, {len(df.columns)} columns")
            return True
        else:
            print(f"  ✗ {path}: Only {len(df)} rows (minimum {min_rows} required)")
            return False
    except Exception as e:
        print(f"  ✗ Error reading {path}: {e}")
        return False

def check_notebook(path: str) -> bool:
    """Check if Jupyter notebook exists and has cells."""
    if not Path(path).exists():
        print(f"  ✗ Notebook not found: {path}")
        return False
    
    try:
        with open(path, 'r') as f:
            nb = json.load(f)
            cell_count = len(nb.get('cells', []))
            if cell_count > 0:
                print(f"  ✓ {path}: {cell_count} cells")
                return True
            else:
                print(f"  ✗ {path}: No cells found")
                return False
    except Exception as e:
        print(f"  ✗ Error reading {path}: {e}")
        return False

def main():
    print("=" * 60)
    print("MILESTONE 2 VERIFICATION CHECKLIST")
    print("=" * 60)
    
    checks = []
    
    # 1. Featurizer
    print("\n1. Featurizer (features/featurizer.py)")
    checks.append(check_file('features/featurizer.py', 'Featurizer script'))
    
    # 2. Replay script
    print("\n2. Replay Script (scripts/replay.py)")
    checks.append(check_file('scripts/replay.py', 'Replay script'))
    
    # 3. Feature data
    print("\n3. Processed Features")
    checks.append(check_parquet_file('data/processed/features.parquet', min_rows=100))
    
    # 4. EDA notebook
    print("\n4. EDA Notebook (notebooks/eda.ipynb)")
    checks.append(check_notebook('notebooks/eda.ipynb'))
    
    # 5. Feature specification
    print("\n5. Feature Specification (docs/feature_spec.md)")
    checks.append(check_file('docs/feature_spec.md', 'Feature spec document'))
    
    # 6. Evidently report
    print("\n6. Evidently Report (reports/evidently/)")
    html_report = check_file('reports/evidently/data_drift_report.html', 'HTML report')
    json_report = check_file('reports/evidently/data_drift_report.json', 'JSON report')
    checks.append(html_report or json_report)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ ALL CHECKS PASSED! Milestone 2 complete.")
    else:
        print(f"✗ {total - passed} checks failed. Review output above.")
    print("=" * 60)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
    
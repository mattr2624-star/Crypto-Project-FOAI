import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")
REFERENCE_PATH = DATA_DIR / "reference.csv"
CURRENT_PATH = DATA_DIR / "current.csv"

N_CURRENT = 200  # number of rows to sample for current window


if __name__ == "__main__":
    print("📥 Loading reference...")
    ref = pd.read_csv(REFERENCE_PATH)

    sample = ref.sample(min(N_CURRENT, len(ref)), random_state=42)

    CURRENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(CURRENT_PATH, index=False)

    print(f"📝 Saving current sample → {CURRENT_PATH} (rows={len(sample)})")
    print("🎉 Current sample generated!")

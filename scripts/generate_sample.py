import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")
INPUT_PATH = DATA_DIR / "btcusd_ticks_10min.csv"
OUTPUT_PATH = DATA_DIR / "crypto_sample.csv"


if __name__ == "__main__":
    print("📥 Loading replay data...")
    df = pd.read_csv(INPUT_PATH)

    # Take a tiny random subset for debugging / examples
    sample = df.sample(min(200, len(df)), random_state=123)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"🔧 Saving sample selection → {OUTPUT_PATH}")
    sample.to_csv(OUTPUT_PATH, index=False)

    print("🎉 Sample CSV generated!")

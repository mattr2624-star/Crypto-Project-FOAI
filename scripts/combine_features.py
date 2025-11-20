import pandas as pd
from pathlib import Path

DATA_DIR = Path("/app/data")
RAW_PATH = DATA_DIR / "btcusd_ticks_10min.csv"
REFERENCE_PATH = DATA_DIR / "reference.csv"


def make_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Very simple toy feature engineering:
    - ret_mean: rolling mean of log returns
    - ret_std: rolling std of log returns
    - n: constant window length
    """
    df = df.sort_values("timestamp")

    df["price"] = df["price"].astype(float)
    df["log_price"] = df["price"].apply(lambda x: float(pd.np.log(x)))  # type: ignore[attr-defined]
    df["log_ret"] = df["log_price"].diff()

    feats = pd.DataFrame()
    feats["ret_mean"] = df["log_ret"].rolling(window).mean()
    feats["ret_std"] = df["log_ret"].rolling(window).std()
    feats["n"] = window

    feats = feats.dropna().reset_index(drop=True)
    return feats


if __name__ == "__main__":
    print("📥 Loading raw price data...")
    df = pd.read_csv(RAW_PATH)

    feats = make_features(df, window=10)

    print(f"📝 Saving feature dataset → {REFERENCE_PATH}")
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(REFERENCE_PATH, index=False)

    print("🎉 Feature file generated!")

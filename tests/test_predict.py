import requests
import time

API = "http://localhost:8000/predict"

sample = {
    "rows": [
        {
            "midprice": 68000.5,
            "spread": 1.2,
            "trade_intensity": 14,
            "volatility_30s": 0.02,
        }
    ]
}


def test_predict():
    """Robust API test with retries and semantic validation."""

    resp = None

    # Allow startup time for container CI
    for _ in range(10):
        try:
            resp = requests.post(API, json=sample, timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)

    assert resp is not None, "No response from API"
    assert resp.status_code == 200, f"Bad status: {resp.status_code}"

    data = resp.json()

    # Contract: Check required fields
    required = ["volatility_score", "model_name", "model_version", "model_variant"]
    for key in required:
        assert key in data, f"Missing key in response: {key}"

    # Behavioral expectations
    assert isinstance(data["volatility_score"], float), "volatility_score must be float"
    assert 0.0 <= data["volatility_score"] <= 1.0, "volatility_score should be [0,1] probability"

    assert data["model_variant"] in ["ml", "baseline", "student1", "student2"], (
        f"Invalid model_variant: {data['model_variant']}"
    )

import requests

def test_predict_endpoint():
    payload = {"rows": [{
        "midprice": 100,
        "spread": 0.5,
        "trade_intensity": 1.2,
        "volatility_30s": 0.00005
    }]}
    res = requests.post("http://localhost:8000/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert "scores" in body

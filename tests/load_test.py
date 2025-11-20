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


def main():
    print("\n🚀 Running 100-Burst Load Test\n")

    latencies = []
    fails = 0

    for _ in range(100):
        start = time.time()
        try:
            r = requests.post(API, json=sample, timeout=1)
            if r.status_code == 200:
                elapsed_ms = (time.time() - start) * 1000
                latencies.append(elapsed_ms)
            else:
                fails += 1
        except Exception:
            fails += 1

    print("📌 Summary:")
    print(f"• Total Requests: 100")
    print(f"• Failures: {fails}")

    if latencies:
        latencies.sort()
        n = len(latencies)
        p50 = latencies[int(0.50 * (n - 1))]
        p95 = latencies[int(0.95 * (n - 1))]
        pmax = max(latencies)

        print("\n⏱️ Latency Stats:")
        print(f"• p50 (median): {p50:.2f} ms")
        print(f"• p95: {p95:.2f} ms")
        print(f"• Max: {pmax:.2f} ms")
    else:
        print("\n⚠️ No success responses → investigate model-server startup or timeout settings.")

    print("\n🎉 Load Test Complete\n")


if __name__ == "__main__":
    main()

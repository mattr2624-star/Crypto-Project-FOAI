import json
import csv
import threading
import time
import websocket

OUTPUT_FILE = r"C:\cp\data\btcusd_ticks_10min.csv"

def on_message(ws, message):
    msg = json.loads(message)

    # We only want "ticker" messages (trade-level data)
    if msg.get("type") != "ticker":
        return

    trade = {
        "timestamp": msg["time"],
        "price": float(msg["price"]),
        "volume": float(msg["last_size"]),
    }

    # Append row to CSV
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trade["timestamp"], trade["price"], trade["volume"]])

    print(trade)


def on_open(ws):
    print("📡 Connected to Coinbase BTC-USD ticker stream...")

    # Subscribe to the ticker channel for BTC-USD
    subscribe_msg = {
        "type": "subscribe",
        "channels": [
            {"name": "ticker", "product_ids": ["BTC-USD"]}
        ],
    }

    ws.send(json.dumps(subscribe_msg))

    # Create CSV with header
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price", "volume"])


def on_error(ws, error):
    print("❌ Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket closed")


def run_for_10_minutes(ws):
    """Run the WebSocket for 10 minutes then close it."""
    time.sleep(60 * 10)
    print("⏱️ 10 minutes complete — closing WebSocket...")
    ws.close()


if __name__ == "__main__":
    socket_url = "wss://ws-feed.exchange.coinbase.com"

    ws = websocket.WebSocketApp(
        socket_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    print("⌛ Collecting LIVE BTC-USD ticks for 10 minutes...")

    # Start WebSocket in background thread
    thread = threading.Thread(target=lambda: ws.run_forever())
    thread.start()

    # Close after 10 minutes
    run_for_10_minutes(ws)

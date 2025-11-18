"""
drift_scheduler.py – Keeps drift monitoring alive and updates periodically
"""
import time
import subprocess

print("🕒 Drift monitor scheduler started (interval = 5 minutes)")
while True:
    subprocess.run(["python", "drift_monitor.py"])
    print("✅ Drift monitor run complete, sleeping 5 minutes...")
    time.sleep(300)

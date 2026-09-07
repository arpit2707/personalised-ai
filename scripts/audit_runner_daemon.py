import time
import sys
import os

# Ensure stdout is unbuffered
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.system_auditor import run_health_audit

INTERVAL_SECONDS = 900  # 15 minutes

def main():
    print(f"[DAEMON] Starting 15-Minute Autonomous Architecture Auditor (interval={INTERVAL_SECONDS}s)...")
    while True:
        try:
            report = run_health_audit()
            status = report.get("overall_status", "UNKNOWN")
            print(f"[DAEMON] Audit complete at {report.get('timestamp')}. Status: {status}")
        except Exception as e:
            print(f"[DAEMON] Error during audit: {e}")
            
        print(f"[DAEMON] Sleeping for {INTERVAL_SECONDS} seconds until next scheduled audit...\n")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()

import time
import sys
import os

# Ensure stdout is unbuffered
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.system_auditor import run_health_audit

AUDIT_INTERVAL_SECONDS = 900  # 15 minutes architecture health audit
TASK_CHECK_INTERVAL_SECONDS = 600  # 10 minutes task progression cross-verification

def main():
    print("=" * 70)
    print("🚀 REEL2REAL AUTONOMOUS AUDITOR & TASK SCHEDULER DAEMON")
    print(f"Full Architecture Audit Cycle: Every {AUDIT_INTERVAL_SECONDS // 60} minutes")
    print(f"Task Cross-Verification Cycle: Every {TASK_CHECK_INTERVAL_SECONDS // 60} minutes")
    print("=" * 70 + "\n")

    last_full_audit = 0
    last_task_check = 0

    while True:
        now = time.time()
        
        # 15-minute full architecture audit
        if now - last_full_audit >= AUDIT_INTERVAL_SECONDS or last_full_audit == 0:
            print(f"\n[DAEMON] Triggering 15-Minute Full System Architecture Audit...")
            try:
                report = run_health_audit()
                last_full_audit = time.time()
                last_task_check = last_full_audit
                print(f"[DAEMON] Audit complete. Overall Status: {report.get('overall_status')} (Score: {report.get('health_score')}/100)")
            except Exception as e:
                print(f"[DAEMON] Full audit encountered an error: {e}")

        # 10-minute task cross-check
        elif now - last_task_check >= TASK_CHECK_INTERVAL_SECONDS:
            print(f"\n[DAEMON] Triggering 10-Minute Task Cross-Verification Cycle...")
            try:
                report = run_health_audit()
                last_task_check = time.time()
                task_board = report.get("task_board", {})
                print(f"[DAEMON] Task check complete. Active task: {task_board.get('in_progress')}")
            except Exception as e:
                print(f"[DAEMON] Task check error: {e}")

        time.sleep(30)

if __name__ == "__main__":
    main()

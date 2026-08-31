"""
run_pipeline.py
Runs the full content personalization pipeline in order.
Usage: python run_pipeline.py
       python run_pipeline.py --skip-dbt        # skip dbt if models already built
       python run_pipeline.py --dashboard-only   # just launch the dashboard
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime

# ── config ────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DBT_DIR     = os.path.join(BASE_DIR, "dbt_code")
PYTHON_DIR  = os.path.join(BASE_DIR, "python")
DASHBOARD   = os.path.join(PYTHON_DIR, "dashboard", "app.py")

STEPS = [
    {
        "name":    "DBT pipeline",
        "command": ["dbt", "run"],
        "cwd":     DBT_DIR,
        "skip_flag": "skip_dbt"
    },
    {
        "name":    "Movie cooccurrence (sparse matrix Pearson)",
        "command": [sys.executable, os.path.join(PYTHON_DIR, "analysis", "cooccurrence.py")],
        "cwd":     BASE_DIR,
        "skip_flag": None
    },
    {
        "name":    "SVD recommendation model",
        "command": [sys.executable, os.path.join(PYTHON_DIR, "modeling", "svd_model.py")],
        "cwd":     BASE_DIR,
        "skip_flag": None
    },
    {
        "name":    "A/B test simulation",
        "command": [sys.executable, os.path.join(PYTHON_DIR, "modeling", "ab_test.py")],
        "cwd":     BASE_DIR,
        "skip_flag": None
    },
    {
        "name":    "Pre-compute recommendations",
        "command": [sys.executable, os.path.join(PYTHON_DIR, "modeling", "precompute.py")],
        "cwd":     BASE_DIR,
        "skip_flag": None
    },
    {
        "name":    "Push to MotherDuck",
        "command": [sys.executable, os.path.join(PYTHON_DIR, "modeling", "push_to_motherduck.py")],
        "cwd":     BASE_DIR,
        "skip_flag": None
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg, color=None):
    colors = {
        "green":  "\033[92m",
        "red":    "\033[91m",
        "yellow": "\033[93m",
        "cyan":   "\033[96m",
        "reset":  "\033[0m"
    }
    prefix = colors.get(color, "") if color else ""
    reset  = colors["reset"] if color else ""
    ts     = datetime.now().strftime("%H:%M:%S")
    print(f"{prefix}[{ts}] {msg}{reset}", flush=True)

def run_step(step):
    log(f"Starting: {step['name']}", "cyan")
    start = datetime.now()

    result = subprocess.run(
        step["command"],
        cwd=step["cwd"],
        capture_output=False,   # stream output live
        text=True
    )

    elapsed = (datetime.now() - start).seconds
    if result.returncode == 0:
        log(f"✓ Completed: {step['name']} ({elapsed}s)", "green")
        return True
    else:
        log(f"✗ Failed: {step['name']} (exit code {result.returncode})", "red")
        return False

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run content personalization pipeline")
    parser.add_argument("--skip-dbt",       action="store_true", help="Skip dbt run")
    parser.add_argument("--dashboard-only", action="store_true", help="Launch dashboard only")
    parser.add_argument("--no-dashboard",   action="store_true", help="Run pipeline but skip dashboard")
    args = parser.parse_args()

    log("=" * 55, "cyan")
    log(" Content Personalization Pipeline", "cyan")
    log("=" * 55, "cyan")

    if args.dashboard_only:
        log("Launching dashboard only...", "yellow")
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", DASHBOARD])
        return

    # run pipeline steps
    failed = []
    for step in STEPS:
        # check skip flags
        if step["skip_flag"] == "skip_dbt" and args.skip_dbt:
            log(f"Skipping: {step['name']}", "yellow")
            continue

        success = run_step(step)
        if not success:
            failed.append(step["name"])
            log(f"Pipeline halted at: {step['name']}", "red")
            log("Fix the error above and re-run. You can use --skip-dbt to skip completed steps.", "yellow")
            sys.exit(1)

    # summary
    log("=" * 55, "cyan")
    log(" Pipeline complete", "green")
    log("=" * 55, "cyan")

    if not args.no_dashboard:
        log("Launching dashboard...", "cyan")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", DASHBOARD
        ])

if __name__ == "__main__":
    main()
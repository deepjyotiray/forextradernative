"""
Deploy Watcher — polls git remote for changes, pulls, and restarts auto_trader.
Run this on the Windows trading machine. It checks every 30s for new commits.
"""
import subprocess
import time
import os
import sys
import logging

POLL_INTERVAL = 30  # seconds
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADER_URL = "http://127.0.0.1:8899"
LOG_FILE = os.path.join(BASE_DIR, "deploy_watcher.log")

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run(cmd, cwd=BASE_DIR):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def has_new_commits():
    run("git fetch origin")
    local, _, _ = run("git rev-parse HEAD")
    branch, _, _ = run("git rev-parse --abbrev-ref HEAD")
    remote, _, rc = run(f"git rev-parse origin/{branch}")
    if rc != 0:
        return False
    return local != remote and bool(remote)


def pull():
    out, err, code = run("git pull --ff-only")
    log.info(f"git pull: {out or err}")
    return code == 0


def stop_trader():
    try:
        import urllib.request
        req = urllib.request.Request(f"{TRADER_URL}/shutdown", method="POST")
        urllib.request.urlopen(req, timeout=5)
        log.info("Sent /shutdown to trader")
        time.sleep(3)
    except Exception:
        log.info("Trader not running or already stopped")


def start_trader():
    vbs = os.path.join(BASE_DIR, "start_trader.vbs")
    if os.path.exists(vbs):
        subprocess.Popen(["wscript", vbs], cwd=BASE_DIR)
        log.info("Started trader via start_trader.vbs")
    else:
        # fallback: run directly
        python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
        if not os.path.exists(python):
            python = sys.executable
        subprocess.Popen(
            [python, "auto_trader.py"],
            cwd=BASE_DIR, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        log.info(f"Started trader via {python}")


def install_deps():
    """pip install if requirements.txt changed in the pull."""
    python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python):
        python = sys.executable
    out, err, code = run(f'"{python}" -m pip install -r requirements.txt -q')
    if code == 0:
        log.info("Dependencies up to date")
    else:
        log.info(f"pip install issue: {err}")


def deploy():
    log.info("New commits detected — deploying...")
    stop_trader()
    if not pull():
        log.info("Pull failed, skipping deploy")
        start_trader()
        return
    install_deps()
    start_trader()
    log.info("Deploy complete ✓")


if __name__ == "__main__":
    log.info("Deploy watcher started")
    print(f"Deploy watcher running. Polling every {POLL_INTERVAL}s. Log: {LOG_FILE}")
    while True:
        try:
            if has_new_commits():
                deploy()
            else:
                log.debug("No changes")
        except Exception as e:
            log.error(f"Error: {e}")
        time.sleep(POLL_INTERVAL)

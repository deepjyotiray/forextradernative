"""
Deploy Watcher — polls git remote for changes, fast-forwards the repo,
and restarts the unified trader on the Windows trading machine.

Recommended workflow:
1. Make changes on another machine
2. Push to the tracked branch on origin
3. This watcher notices the new commit, pulls, refreshes deps, and restarts

Run this on the Windows trading machine, ideally as a Scheduled Task at logon.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

POLL_INTERVAL = int(os.getenv("DEPLOY_WATCHER_POLL_INTERVAL", "30") or "30")
BASE_DIR = Path(__file__).resolve().parent
TRADER_URLS = [
    "http://127.0.0.1:8000",  # current unified stack
    "http://127.0.0.1:8899",  # legacy single-service fallback
]
LOG_FILE = BASE_DIR / "deploy_watcher.log"

try:
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
except PermissionError:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.warning("Could not open deploy_watcher.log, logging to console only")
log = logging.getLogger(__name__)


def run(cmd: str, cwd: Path = BASE_DIR) -> tuple[str, str, int]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        shell=True,
    )
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


def git_branch() -> str:
    branch, _, rc = run("git rev-parse --abbrev-ref HEAD")
    if rc != 0 or not branch:
        raise RuntimeError("Unable to determine current git branch")
    return branch


def has_new_commits() -> bool:
    fetch_out, fetch_err, fetch_rc = run("git fetch origin")
    if fetch_rc != 0:
        log.warning(f"git fetch failed: {fetch_err or fetch_out}")
        return False
    branch = git_branch()
    local, _, local_rc = run("git rev-parse HEAD")
    remote, _, remote_rc = run(f"git rev-parse origin/{branch}")
    if local_rc != 0 or remote_rc != 0:
        return False
    return bool(local and remote and local != remote)


def pull() -> bool:
    out, err, code = run("git pull --ff-only")
    log.info(f"git pull: {out or err or 'ok'}")
    return code == 0


def install_deps() -> None:
    python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    out, err, code = run(f'"{python}" -m pip install -r requirements.txt -q')
    if code == 0:
        log.info("Dependencies up to date")
    else:
        log.warning(f"pip install issue: {err or out}")


def stop_trader() -> None:
    stopped = False
    for base_url in TRADER_URLS:
        try:
            req = urllib.request.Request(f"{base_url}/shutdown", method="POST")
            urllib.request.urlopen(req, timeout=5)
            log.info(f"Sent /shutdown to trader at {base_url}")
            stopped = True
            time.sleep(3)
        except urllib.error.URLError:
            continue
        except Exception as exc:
            log.info(f"Shutdown request to {base_url} failed: {exc}")
    if not stopped:
        log.info("Trader API not running; falling back to process cleanup")

    # Clean up any unified startup processes that outlived the API shutdown.
    ps_script = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.CommandLine -like '*unified_startup.py*' -or $_.Name -eq 'AutoTrader.exe' } "
        "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def start_trader() -> None:
    launcher_exe = BASE_DIR / "UnifiedTraderRestart.exe"
    startup_script = BASE_DIR / "unified_startup.py"
    if launcher_exe.exists():
        subprocess.Popen(
            [str(launcher_exe)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log.info("Started trader via UnifiedTraderRestart.exe")
        return

    python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    subprocess.Popen(
        [str(python), "-u", str(startup_script)],
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    log.info(f"Started trader via {python.name} unified_startup.py")


def deploy() -> None:
    log.info("New commits detected — deploying...")
    stop_trader()
    if not pull():
        log.warning("Pull failed, attempting to restart current code")
        start_trader()
        return
    install_deps()
    start_trader()
    log.info("Deploy complete")


if __name__ == "__main__":
    log.info("Deploy watcher started")
    print(f"Deploy watcher running. Polling every {POLL_INTERVAL}s. Log: {LOG_FILE}")
    while True:
        try:
            if has_new_commits():
                deploy()
        except Exception as exc:
            log.error(f"Error: {exc}")
        time.sleep(POLL_INTERVAL)

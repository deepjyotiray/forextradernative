import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
import ctypes
import urllib.error
import urllib.request


POLL_INTERVAL = int(os.getenv("DEPLOY_WATCHER_POLL_INTERVAL", "30") or "30")
TRADER_URLS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8899",
]


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _stop_existing_unified() -> None:
    ps_script = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.CommandLine -like '*unified_startup.py*' } "
        "| ForEach-Object { "
        "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue "
        "}"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _run_shell(cmd: str, cwd: str) -> tuple[str, str, int]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        shell=True,
    )
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


def _git_branch(base_dir: str) -> str | None:
    branch, _, rc = _run_shell("git rev-parse --abbrev-ref HEAD", base_dir)
    if rc != 0 or not branch:
        return None
    return branch


def _has_new_commits(base_dir: str) -> tuple[bool, str]:
    fetch_out, fetch_err, fetch_rc = _run_shell("git fetch origin", base_dir)
    if fetch_rc != 0:
        return False, fetch_err or fetch_out or "git fetch failed"

    branch = _git_branch(base_dir)
    if not branch:
        return False, "unable to determine current branch"

    local, _, local_rc = _run_shell("git rev-parse HEAD", base_dir)
    remote, _, remote_rc = _run_shell(f"git rev-parse origin/{branch}", base_dir)
    if local_rc != 0 or remote_rc != 0 or not local or not remote:
        return False, f"unable to compare local HEAD to origin/{branch}"
    if local == remote:
        return False, ""
    return True, f"new commit detected on {branch}: {local[:7]} -> {remote[:7]}"


def _pull_latest(base_dir: str) -> tuple[bool, str]:
    out, err, rc = _run_shell("git pull --ff-only", base_dir)
    return rc == 0, out or err or "git pull completed"


def _install_deps(base_dir: str) -> tuple[bool, str]:
    python = pathlib.Path(base_dir) / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = pathlib.Path(sys.executable)
    out, err, rc = _run_shell(f'"{python}" -m pip install -r requirements.txt -q', base_dir)
    return rc == 0, out or err or "dependencies up to date"


def _request_shutdown() -> bool:
    for base_url in TRADER_URLS:
        try:
            req = urllib.request.Request(f"{base_url}/shutdown", method="POST")
            urllib.request.urlopen(req, timeout=5)
            return True
        except urllib.error.URLError:
            continue
        except Exception:
            continue
    return False


def _start_unified(base_dir: str) -> subprocess.Popen | None:
    startup_script = os.path.join(base_dir, "unified_startup.py")
    if not os.path.exists(startup_script):
        return None

    create_no_window = 0x08000000

    return subprocess.Popen(
        ["py", "-u", startup_script],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=create_no_window,
    )


class UnifiedTraderWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.base_dir = _base_dir()
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.log_tail_stop = threading.Event()
        self.watcher_enabled = threading.Event()
        self.deploy_lock = threading.Lock()
        self.watcher_thread: threading.Thread | None = None

        self.root.title("Unified Trader Console")
        self.root.geometry("980x640")
        self.root.minsize(760, 500)

        ico_path = os.path.join(self.base_dir, "UnifiedTraderRestart.ico")
        if os.path.exists(ico_path):
            try:
                self.root.iconbitmap(ico_path)
            except Exception:
                pass

        top_bar = tk.Frame(self.root)
        top_bar.pack(fill="x", padx=10, pady=(10, 6))

        self.status_var = tk.StringVar(value="Initializing...")
        status_label = tk.Label(top_bar, textvariable=self.status_var, anchor="w")
        status_label.pack(side="left", fill="x", expand=True)

        self.watcher_var = tk.StringVar(value=f"Auto Deploy: On ({POLL_INTERVAL}s)")
        watcher_label = tk.Label(top_bar, textvariable=self.watcher_var, anchor="e")
        watcher_label.pack(side="right", padx=(10, 0))

        watcher_btn = tk.Button(top_bar, text="Pause Auto Deploy", command=self.toggle_watcher)
        watcher_btn.pack(side="right", padx=(6, 0))
        self.watcher_btn = watcher_btn

        restart_btn = tk.Button(top_bar, text="Restart", command=self.restart_app)
        restart_btn.pack(side="right", padx=(6, 0))

        stop_btn = tk.Button(top_bar, text="Stop", command=self.stop_app)
        stop_btn.pack(side="right", padx=(6, 0))

        clear_btn = tk.Button(top_bar, text="Clear", command=self.clear_logs)
        clear_btn.pack(side="right")

        self.log_view = scrolledtext.ScrolledText(
            self.root,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
        )
        self.log_view.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._drain_queue)
        self.watcher_enabled.set()
        self.restart_app()
        self._start_watcher()

    def _append(self, line: str) -> None:
        self.log_view.configure(state="normal")
        self.log_view.insert("end", line.rstrip("\n") + "\n")
        self.log_view.see("end")
        self.log_view.configure(state="disabled")

    def _drain_queue(self) -> None:
        try:
            while True:
                self._append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        if not self.shutdown_event.is_set():
            self.root.after(100, self._drain_queue)

    def _enqueue(self, line: str) -> None:
        self.log_queue.put(line)

    def _set_status(self, value: str) -> None:
        self.root.after(0, lambda: self.status_var.set(value))

    def _set_watcher_status(self, value: str) -> None:
        self.root.after(0, lambda: self.watcher_var.set(value))

    def _stream_process_output(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            if self.shutdown_event.is_set():
                return
            self._enqueue(line.rstrip("\n"))
        proc.wait()
        self._enqueue(f"[launcher] unified process exited with code {proc.returncode}")
        self._set_status(f"Stopped (exit code {proc.returncode})")

    def _tail_trader_log(self) -> None:
        log_path = os.path.join(self.base_dir, "trader.log")
        position = 0
        if os.path.exists(log_path):
            try:
                position = os.path.getsize(log_path)
            except OSError:
                position = 0

        while not self.log_tail_stop.is_set():
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(position)
                        data = f.read()
                        position = f.tell()
                        if data:
                            for line in data.splitlines():
                                self._enqueue(f"[trader.log] {line}")
                except OSError:
                    pass
            time.sleep(0.4)

    def restart_app(self) -> None:
        self._enqueue("[launcher] restarting unified trader...")
        self._set_status("Restarting...")
        self.stop_app(silent=True)
        _stop_existing_unified()
        time.sleep(1.2)

        proc = _start_unified(self.base_dir)
        if proc is None:
            self._set_status("Failed: unified_startup.py missing")
            messagebox.showerror(
                "Unified Trader",
                "Could not find unified_startup.py next to this executable.",
            )
            return

        self.process = proc
        self._enqueue("[launcher] started unified_startup.py")
        self._enqueue("[launcher] dashboard: http://localhost:8000/dashboard")
        self._set_status("Running")

        threading.Thread(
            target=self._stream_process_output,
            args=(proc,),
            daemon=True,
            name="UnifiedStdoutReader",
        ).start()

        self.log_tail_stop.clear()
        threading.Thread(
            target=self._tail_trader_log,
            daemon=True,
            name="TraderLogTail",
        ).start()

    def stop_app(self, silent: bool = False) -> None:
        self.log_tail_stop.set()
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=4)
            except Exception:
                _stop_existing_unified()
            if not silent:
                self._enqueue("[launcher] stop signal sent")
        self.process = None
        if not silent:
            self._set_status("Stopped")

    def _start_watcher(self) -> None:
        if self.watcher_thread and self.watcher_thread.is_alive():
            return
        self.watcher_thread = threading.Thread(
            target=self._watch_for_deploys,
            daemon=True,
            name="DeployWatcher",
        )
        self.watcher_thread.start()
        self._enqueue(f"[watcher] auto deploy enabled; polling every {POLL_INTERVAL}s")

    def toggle_watcher(self) -> None:
        if self.watcher_enabled.is_set():
            self.watcher_enabled.clear()
            self.watcher_btn.configure(text="Resume Auto Deploy")
            self._set_watcher_status("Auto Deploy: Paused")
            self._enqueue("[watcher] auto deploy paused")
            return

        self.watcher_enabled.set()
        self.watcher_btn.configure(text="Pause Auto Deploy")
        self._set_watcher_status(f"Auto Deploy: On ({POLL_INTERVAL}s)")
        self._enqueue("[watcher] auto deploy resumed")
        self._start_watcher()

    def _deploy_latest(self) -> None:
        if not self.deploy_lock.acquire(blocking=False):
            return

        try:
            self._enqueue("[watcher] new commit found; deploying latest code...")
            self._set_status("Deploying latest code...")

            if _request_shutdown():
                self._enqueue("[watcher] shutdown request sent to trader API")
                time.sleep(3)
            else:
                self._enqueue("[watcher] trader API not reachable; using local process stop")

            self.stop_app(silent=True)
            _stop_existing_unified()
            time.sleep(1.2)

            pulled, pull_msg = _pull_latest(self.base_dir)
            self._enqueue(f"[watcher] {pull_msg}")
            if not pulled:
                self._enqueue("[watcher] pull failed; restarting current local code")
                self.restart_app()
                return

            deps_ok, deps_msg = _install_deps(self.base_dir)
            self._enqueue(f"[watcher] {deps_msg}")
            if not deps_ok:
                self._enqueue("[watcher] dependency refresh reported an issue; continuing restart")

            self.restart_app()
            self._enqueue("[watcher] deploy complete")
        finally:
            self.deploy_lock.release()

    def _watch_for_deploys(self) -> None:
        while not self.shutdown_event.is_set():
            if not self.watcher_enabled.is_set():
                time.sleep(1)
                continue

            try:
                changed, detail = _has_new_commits(self.base_dir)
                if changed:
                    if detail:
                        self._enqueue(f"[watcher] {detail}")
                    self._deploy_latest()
                elif detail:
                    self._enqueue(f"[watcher] {detail}")
            except Exception as exc:
                self._enqueue(f"[watcher] error: {exc}")

            for _ in range(POLL_INTERVAL):
                if self.shutdown_event.is_set():
                    return
                time.sleep(1)

    def clear_logs(self) -> None:
        self.log_view.configure(state="normal")
        self.log_view.delete("1.0", "end")
        self.log_view.configure(state="disabled")

    def on_close(self) -> None:
        self.shutdown_event.set()
        self.watcher_enabled.clear()
        self.log_tail_stop.set()
        self.root.destroy()


def main() -> None:
    # Give Windows a stable identity so pinning/taskbar icon does not fall back.
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ForexTraderNative.UnifiedTraderRestart"
        )
    except Exception:
        pass

    root = tk.Tk()
    UnifiedTraderWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

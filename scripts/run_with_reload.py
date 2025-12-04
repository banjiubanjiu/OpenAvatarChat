#!/usr/bin/env python3
"""
Lightweight hot-reload runner.

Usage:
  python scripts/run_with_reload.py --watch src --watch config -- uv run src/demo.py --config config/chat_with_lam_qwen3_asr.yaml

It will start the command, watch the given paths for mtime changes, and restart the command on change.
Dependencies: standard library only.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Sequence, Tuple


def parse_args() -> Tuple[List[str], List[str], float]:
    parser = argparse.ArgumentParser(description="Run a command with simple hot-reload on file changes.")
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        help="Path to watch (can be given multiple times). Defaults to src and config.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run (prefix with -- to separate).",
    )
    args = parser.parse_args()
    watch_paths = args.watch or ["src", "config"]
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("No command provided. Example: -- uv run src/demo.py --config config/chat_with_lam_qwen3_asr.yaml")
    return watch_paths, cmd, args.interval


def latest_mtime(paths: Sequence[str]) -> float:
    latest = 0.0
    for root_path in paths:
        if not os.path.exists(root_path):
            continue
        if os.path.isfile(root_path):
            latest = max(latest, os.path.getmtime(root_path))
            continue
        for dirpath, _, filenames in os.walk(root_path):
            for name in filenames:
                fp = os.path.join(dirpath, name)
                try:
                    latest = max(latest, os.path.getmtime(fp))
                except OSError:
                    continue
    return latest


def restart_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main() -> int:
    watch_paths, cmd, interval = parse_args()
    print(f"[reload] watching: {watch_paths}")
    print(f"[reload] command: {' '.join(cmd)}")

    mt = latest_mtime(watch_paths)
    proc = subprocess.Popen(cmd)
    try:
        while True:
            time.sleep(interval)
            new_mt = latest_mtime(watch_paths)
            # If the process exited on its own, break to avoid silent restart loops.
            if proc.poll() is not None:
                print(f"[reload] process exited with code {proc.returncode}")
                return proc.returncode or 0
            if new_mt > mt:
                print("[reload] change detected, restarting...")
                mt = new_mt
                restart_process(proc)
                proc = subprocess.Popen(cmd)
    except KeyboardInterrupt:
        print("[reload] stopping...")
        restart_process(proc)
        return 0


if __name__ == "__main__":
    sys.exit(main())

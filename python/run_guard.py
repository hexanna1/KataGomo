"""Hold one training lock across the launcher and its descendants."""

import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def terminal_launchers(script_name, exclude=()):
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,state=,command="],
        capture_output=True, text=True, check=True,
    )
    candidates = {}
    pattern = re.compile(r"(?:^|\s)(?:\S*/)?" + re.escape(script_name) + r"(?:\s|$)")
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        pid, pgid, state, command = fields
        pid = int(pid)
        if pid in exclude or state.startswith("Z"):
            continue
        if Path(command.split()[0]).name not in ("bash", "sh", "zsh"):
            continue
        if pattern.search(command):
            candidates[pid] = {"pid": pid, "pgid": int(pgid)}
    if not candidates:
        return {}
    result = subprocess.run(
        ["lsof", "-a", "-p", ",".join(map(str, candidates)), "-d", "cwd", "-Fn"],
        capture_output=True, text=True,
    )
    found = {}
    pid = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("n") and pid in candidates:
            found.setdefault(Path(line[1:]).resolve(), []).append(candidates[pid])
    return found


def terminal_runs(script, exclude=()):
    script = Path(script).resolve()
    return terminal_launchers(script.name, exclude).get(script.parent, [])


def read_lock(data_dir):
    path = Path(data_dir) / "training.lock"
    if not path.exists():
        return False, {}
    with path.open() as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        else:
            locked = False
            fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            info = json.load(handle)
        except json.JSONDecodeError:
            info = {}
    return locked, info


def main():
    data_dir = Path(sys.argv[1]).resolve()
    script = Path(sys.argv[2]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = (data_dir / "training.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Training is already running in " + str(data_dir))
    info = {
        "pid": os.getpid(), "pgid": os.getpgrp(),
        "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": os.environ.get("TRAIN_RUN_SOURCE", "terminal"),
    }
    handle.seek(0)
    handle.truncate()
    json.dump(info, handle)
    handle.flush()
    os.set_inheritable(handle.fileno(), True)
    env = dict(os.environ, TRAIN_RUN_LOCK_PID=str(os.getpid()))
    os.execve("/bin/bash", ["/bin/bash", str(script), *sys.argv[3:]], env)


if __name__ == "__main__":
    main()

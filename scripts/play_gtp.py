"""GTP transport and analysis synchronization shared by the board GUIs."""

import re
import subprocess
import threading


RESPONSE = re.compile(r"^([=?])(\d+)(?:\s+.*)?$")


class GTPClient:
    def __init__(self, command, interval_cs, analysis_wide_root_noise, analysis_parser):
        self.interval_cs = interval_cs
        self.analysis_wide_root_noise = analysis_wide_root_noise
        self.analysis_wide_root_noise_sent = None
        self.analysis_parser = analysis_parser
        self.analysis = []
        self.lock = threading.RLock()
        self.next_command_id = 1
        self.analysis_request = None
        self.analysis_ready = False
        self.closed = False
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def send(self, command):
        with self.lock:
            if self.closed or self.proc.poll() is not None:
                return None
            command_id = self.next_command_id
            self.next_command_id += 1
            try:
                self.proc.stdin.write(f"{command_id} {command}\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                return None
            return command_id

    def _read_stdout(self):
        for line in self.proc.stdout:
            response = RESPONSE.match(line.strip())
            if response:
                status, command_id = response.groups()
                with self.lock:
                    if int(command_id) == self.analysis_request:
                        self.analysis_ready = status == "="
                        if status == "?":
                            self.analysis_request = None
                continue
            with self.lock:
                if self.analysis_request is not None and self.analysis_ready:
                    parsed = self.analysis_parser(line)
                    if parsed:
                        self.analysis = parsed

    def set_analysis_wide_root_noise(self, value):
        with self.lock:
            self.analysis_wide_root_noise = value
            if self.analysis_wide_root_noise_sent != value:
                self.send(f"kata-set-param analysisWideRootNoise {value:g}")
                self.analysis_wide_root_noise_sent = value

    def start_analysis(self, player, clear=True):
        with self.lock:
            self.set_analysis_wide_root_noise(self.analysis_wide_root_noise)
            if clear:
                self.analysis = []
            self.analysis_ready = False
            self.analysis_request = self.send(f"kata-analyze {player} {self.interval_cs}")

    def stop_analysis(self, clear=False):
        with self.lock:
            self.analysis_request = None
            self.analysis_ready = False
            if clear:
                self.analysis = []
            self.send("stop")

    def get_analysis(self):
        with self.lock:
            return list(self.analysis)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.stop_analysis(clear=True)
            self.send("quit")
            self.closed = True
            self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

"""Per-model llama-server lifecycle.

llama.cpp has no in-process model swap (unlike oMLX), so a "swap" is just stop+start.
Each call to start() spawns a fresh ``llama-server`` subprocess, waits for /health,
and returns once the server is responsive.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx


DEFAULT_BIN = Path("~/code/llama.cpp/build/bin/llama-server").expanduser()
DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"
HEALTH_TIMEOUT_S = 120.0
SHUTDOWN_GRACE_S = 10.0


@dataclasses.dataclass
class ServerSpec:
    """Static configuration for one llama-server invocation.

    Maps 1:1 onto llama-server CLI flags. Keep field names stable — they're
    the contract for per-model YAML configs.
    """

    model_path: Path
    n_ctx: int = 8192
    n_gpu_layers: int = 99
    n_threads: int = 6
    n_batch: int = 512
    n_ubatch: int = 512
    flash_attn: str = "auto"
    mmap: bool = True
    mlock: bool = False
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    chat_template: str | None = None
    chat_template_file: Path | None = None
    jinja: bool = True
    alias: str | None = None
    extra_args: list[str] = dataclasses.field(default_factory=list)


def _build_argv(
    spec: ServerSpec, host: str, port: int, bin_path: Path, n_parallel: int = 1,
) -> list[str]:
    argv: list[str] = [str(bin_path), "--host", host, "--port", str(port)]
    argv += ["-m", str(spec.model_path)]
    argv += ["-c", str(spec.n_ctx)]
    argv += ["-ngl", str(spec.n_gpu_layers)]
    argv += ["-t", str(spec.n_threads)]
    argv += ["-b", str(spec.n_batch)]
    argv += ["-ub", str(spec.n_ubatch)]
    if n_parallel > 1:
        argv += ["--parallel", str(n_parallel)]
    argv += ["-fa", spec.flash_attn]
    argv += ["-ctk", spec.cache_type_k, "-ctv", spec.cache_type_v]
    if not spec.mmap:
        argv += ["--no-mmap"]
    if spec.mlock:
        argv += ["--mlock"]
    if spec.jinja:
        argv += ["--jinja"]
    else:
        argv += ["--no-jinja"]
    if spec.chat_template:
        argv += ["--chat-template", spec.chat_template]
    if spec.chat_template_file:
        argv += ["--chat-template-file", str(spec.chat_template_file)]
    if spec.alias:
        argv += ["--alias", spec.alias]
    argv += list(spec.extra_args)
    return argv


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def _wait_for_port_free(host: str, port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        if not _port_in_use(host, port):
            return True
        time.sleep(0.2)
    return False


class LlamaServer:
    """Manages one llama-server subprocess.

    Use as a context manager for guaranteed shutdown:

        with LlamaServer(spec).run() as srv:
            ...  # srv.base_url is ready
    """

    def __init__(
        self,
        spec: ServerSpec,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        bin_path: Path = DEFAULT_BIN,
        log_dir: Path | None = None,
        n_parallel: int = 1,
    ) -> None:
        self.spec = spec
        self.host = host
        self.port = port
        self.bin_path = bin_path
        self.n_parallel = n_parallel
        self.log_dir = log_dir or Path.home() / ".llamabench" / "server-logs"
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fp = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("server already started")
        if _port_in_use(self.host, self.port):
            raise RuntimeError(
                f"port {self.port} already in use — another llama-server (or unrelated "
                "service) is bound. Run `llamabench server stop` or change the port."
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        alias = self.spec.alias or self.spec.model_path.stem
        log_path = self.log_dir / f"{ts}-{alias}.log"
        self._log_fp = open(log_path, "wb")
        argv = _build_argv(self.spec, self.host, self.port, self.bin_path, self.n_parallel)
        self._proc = subprocess.Popen(
            argv,
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        self._wait_until_healthy()

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                raise RuntimeError(
                    f"llama-server exited with code {code} during startup. "
                    f"See log for details."
                )
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except (httpx.HTTPError, ValueError) as e:
                last_err = e
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy within {HEALTH_TIMEOUT_S}s "
            f"(last error: {last_err!r})"
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._proc.wait(timeout=SHUTDOWN_GRACE_S)
        self._proc = None
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None
        _wait_for_port_free(self.host, self.port, time.monotonic() + SHUTDOWN_GRACE_S)

    @contextmanager
    def run(self) -> Iterator[LlamaServer]:
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def swap(self, new_spec: ServerSpec) -> None:
        """Stop current model and start a new one on the same port."""
        self.stop()
        self.spec = new_spec
        self.start()

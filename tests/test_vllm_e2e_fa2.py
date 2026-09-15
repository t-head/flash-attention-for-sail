"""End-to-end FA2 interface tests driven by a live vLLM server.

STATUS: two cases are wired here, ``qwen3-32b`` and ``qwen3-coder-30b``, both
run at tp=2 in the ``lqz-vllm`` container (vLLM 0.20.1 PPU image). Each boots a
server, logs ``Using FlashAttention version 2``, and returns a well-formed
``/v1/completions`` response.

Unlike the kernel-level tests in this directory, these cases launch
``vllm serve`` with a real model whose attention is routed to FA2, send a
four-turn conversation with accumulated context, and verify each response.

Why vLLM (and not SGLang) for FA2: on PPU, SGLang's ``fa3`` backend is wired to
the FA3 wheel (``flash_attn_3`` / ``flash_attn_interface``) and never reaches
this repo's FA2 wheel — its only FA2 fallback is gated on
``not _is_fa3_supported()``, which is False on PPU. vLLM's PPU build, by
contrast, binds both wheels in
``vllm/vllm_flash_attn/flash_attn_interface.py``::

    import flash_attn        -> FA2_AVAILABLE   (this repo's root wheel)
    import flash_attn_3._C   -> FA3_AVAILABLE   (this repo's hopper wheel)

and ``--attention-config.flash_attn_version=2`` selects FA2, so the call path is
``vllm ... -> torch.ops.flash_attn._flash_attn_varlen_forward ->
flash_attn/flash_attn_interface.py -> flash_attn_gpu.varlen_fwd``, i.e. exactly
the wheel built by ``build-fa2.sh``. The FA3 counterpart of this file is
``hopper/test_sglang_e2e_fa3.py``.

Covered FA2 entry points (on PPU):

- ``flash_attn_varlen_func`` / ``_flash_attn_varlen_forward`` — every case:
  paged-KV prefill and decode through vLLM's FLASH_ATTN backend, dense GQA at
  head_dim 128, under CUDA graphs.

Each case additionally asserts the server log contains
``Using FlashAttention version 2``, so a silent fallback to FA3 (the PPU
default) fails the test instead of passing quietly.

Two PPU-specific constraints are baked into the launch args:

- ``--block-size 256`` is mandatory: the FA2 kernel enforces
  ``page_block_size % 256 == 0`` (``csrc/flash_attn/flash_api.cpp``), while
  vLLM defaults to 16.
- Models must not be hybrid (Mamba/linear-attention). For hybrid models vLLM
  restricts FA kernel block sizes to ``[16, 32, 64]``
  (``get_supported_kernel_block_sizes`` in ``v1/attention/backends/
  flash_attn.py``), none of which satisfy the %256 rule, so Qwen3.5/Qwen3.8 —
  which are hybrid — cannot currently run FA2 on PPU. Hence the dense Qwen3
  models below.

Every case fits 2 devices of 96 GB.

Usage:

    # Run all cases with pytest (each case boots its own server):
    python -m pytest tests/test_vllm_e2e_fa2.py -v -s

    # Or run directly as a script (exit code 0 = pass, non-zero = fail):
    python tests/test_vllm_e2e_fa2.py                    # all cases
    python tests/test_vllm_e2e_fa2.py --case qwen3-32b   # single case

Environment overrides:

    FA2_E2E_QWEN3_32B_PATH   model path for the qwen3-32b case
    FA2_E2E_QWEN3_CODER_PATH model path for the qwen3-coder-30b case
    FA2_E2E_PORT             server port (default 8997; 0 selects a free port)
    FA2_E2E_TIMEOUT          server startup timeout in seconds (default 1800)
    FA2_E2E_LOG_DIR          directory for server logs (default tests/logs)

Any failure prints the reason plus the tail of the server log and exits
non-zero (or fails the pytest case).

Four requests carry an accumulated User/Assistant transcript through the native
generation API. Assertions cover response structure, successful finish status
and a positive token count, not semantic accuracy. Length-limited generation is
accepted; wording and reasoning length vary between models.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

_TEST_DIR = Path(__file__).resolve().parent
_REPO_DIR = _TEST_DIR.parent
_LOG_DIR = Path(os.environ.get("FA2_E2E_LOG_DIR", _TEST_DIR / "logs"))

_DEFAULT_QWEN3_32B_PATH = "/ppusw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-32B"
_DEFAULT_QWEN3_CODER_PATH = (
    "/ppusw/datasets/checkpoints/LLM/qwen/v3/Qwen3-Coder-30B-A3B-Instruct"
)

_STARTUP_TIMEOUT = int(os.environ.get("FA2_E2E_TIMEOUT", "1800"))
_REQUEST_TIMEOUT = 120
_LOG_TAIL_LINES = 100
_TERM_GRACE = 30
_KILL_GRACE = 15

# Emitted by vllm/v1/attention/backends/flash_attn.py once the FA version is
# resolved; the whole point of these cases is that it says 2 and not 3.
_FA2_LOG_MARKER = "Using FlashAttention version 2"


@dataclasses.dataclass(frozen=True)
class E2ECase:
    """One server-launch test case."""

    name: str
    model_path: str
    # Name the server registers the model under; also used in the request body.
    served_name: str
    # Extra CLI args appended after the common ones.
    extra_args: tuple = ()
    # Human-readable note on which FA2 interfaces this case exercises.
    covers: str = ""
    # Common launch args whose value differs between cases.
    tp_size: int = 2
    gpu_memory_utilization: str = "0.8"
    max_model_len: str = "4096"
    # Extra env vars for the server process, merged over os.environ
    # (case values take precedence).
    env: dict = dataclasses.field(default_factory=dict)


_CASES = {
    "qwen3-32b": E2ECase(
        name="qwen3-32b",
        model_path=os.environ.get(
            "FA2_E2E_QWEN3_32B_PATH", _DEFAULT_QWEN3_32B_PATH
        ),
        served_name="Qwen3-32B",
        covers=(
            "_flash_attn_varlen_forward (paged-KV prefill + decode, dense GQA "
            "64 heads / 8 kv heads, head_dim 128, 64 layers)"
        ),
    ),
    "qwen3-coder-30b": E2ECase(
        name="qwen3-coder-30b",
        model_path=os.environ.get(
            "FA2_E2E_QWEN3_CODER_PATH", _DEFAULT_QWEN3_CODER_PATH
        ),
        served_name="Qwen3-Coder-30B-A3B-Instruct",
        covers=(
            "_flash_attn_varlen_forward (paged-KV prefill + decode, dense GQA "
            "32 heads / 4 kv heads, head_dim 128, 48-layer MoE) — non-hybrid "
            "so the %256 block-size path applies"
        ),
    ),
}


class E2ETestError(RuntimeError):
    """Raised on any e2e failure; carries the full diagnostic message."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_fa2_wheel_is_local_build():
    """Fail unless the installed ``flash_attn`` is this repo's own build.

    The PPU vLLM image ships its own FA wheels, so a case can pass on the
    image's kernels while proving nothing about local changes. pip records the
    install source in ``direct_url.json``; comparing that hash against the
    wheels in ``dist/`` is what distinguishes the two.
    """
    # Resolved in a child interpreter started outside the repo, which is how
    # the `vllm serve` subprocess resolves it. Doing this in-process would
    # instead find the in-tree ./flash_attn source directory whenever the
    # harness itself runs from the repo root (e.g. under pytest).
    probe = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util as u; s = u.find_spec('flash_attn'); "
         "print(s.origin if s else '')"],
        cwd="/", capture_output=True, text=True,
    )
    origin = probe.stdout.strip()
    if probe.returncode != 0 or not origin:
        raise E2ETestError(
            "[E2E FAIL] stage=preflight\n"
            "reason: flash_attn is not importable; vLLM's FA2 path would be "
            "unavailable. Build and install "
            f"{_REPO_DIR}/dist/flash_attn-*.whl first.\n"
            f"probe stderr: {probe.stderr.strip()}"
        )

    pkg_dir = Path(origin).resolve().parent
    dist_infos = sorted(pkg_dir.parent.glob("flash_attn-*.dist-info"))
    if not dist_infos:
        raise E2ETestError(
            f"[E2E FAIL] stage=preflight\n"
            f"reason: no flash_attn-*.dist-info next to {pkg_dir}, so the "
            f"install source cannot be verified."
        )

    installed_url = None
    installed_hash = None
    direct_url = dist_infos[-1] / "direct_url.json"
    if direct_url.is_file():
        info = json.loads(direct_url.read_text())
        installed_url = info.get("url")
        installed_hash = (
            info.get("archive_info", {}).get("hash", "").split("=", 1)[-1]
        )

    local_wheels = sorted((_REPO_DIR / "dist").glob("flash_attn-*.whl"))
    local_hashes = {_sha256(w): w for w in local_wheels}

    if installed_hash not in local_hashes:
        raise E2ETestError(
            f"[E2E FAIL] stage=preflight\n"
            f"reason: the installed flash_attn ({pkg_dir}) did not come from a "
            f"wheel in {_REPO_DIR / 'dist'}, so these cases would exercise "
            f"someone else's FA2 kernels.\n"
            f"  installed from: {installed_url} (sha256={installed_hash})\n"
            f"  local wheels:   "
            f"{ {w.name: h[:12] for h, w in local_hashes.items()} or '<none built>'}\n"
            f"fix: pip install --force-reinstall "
            f"{_REPO_DIR}/dist/flash_attn-*.whl"
        )

    print(
        f"[preflight] flash_attn from {local_hashes[installed_hash].name} "
        f"(sha256={installed_hash[:12]}...), package at {pkg_dir}",
        flush=True,
    )


def _build_cmd(case: E2ECase, port: int) -> list:
    return [
        "vllm",
        "serve",
        case.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(case.tp_size),
        "--gpu-memory-utilization",
        case.gpu_memory_utilization,
        "--max-model-len",
        case.max_model_len,
        # Mandatory on PPU: the FA2 kernel checks page_block_size % 256 == 0.
        "--block-size",
        "256",
        "--trust-remote-code",
        "--served-model-name",
        case.served_name,
        "--attention-backend",
        "FLASH_ATTN",
        # The knob under test: force FA2 instead of the PPU default (FA3).
        "--attention-config.flash_attn_version=2",
        *case.extra_args,
    ]


def _read_log_tail(log_path: Path) -> str:
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"<cannot read log file {log_path}: {e}>"
    tail = "".join(lines[-_LOG_TAIL_LINES:])
    return tail if tail.strip() else "<log file is empty>"


def _fail(case: E2ECase, stage: str, reason: str, log_path: Path):
    message = (
        f"\n{'=' * 72}\n"
        f"[E2E FAIL] case={case.name} stage={stage}\n"
        f"reason: {reason}\n"
        f"server log: {log_path}\n"
        f"----- server log tail (last {_LOG_TAIL_LINES} lines) -----\n"
        f"{_read_log_tail(log_path)}\n"
        f"{'=' * 72}"
    )
    raise E2ETestError(message)



def _select_port(port: int) -> int:
    """Fail early for an occupied explicit port; port 0 asks the OS to choose."""
    if not 0 <= port <= 65535:
        raise E2ETestError(f"invalid port: {port}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Match server binding semantics: a previous case's TIME_WAIT
            # connections must not look like an occupied listening port.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
            return sock.getsockname()[1]
    except OSError as e:
        raise E2ETestError(f"port {port} is unavailable: {e}; use --port 0") from e


def _server_processes(proc):
    """Include reparented workers in the session created by start_new_session."""
    members = []
    for child in psutil.process_iter():
        try:
            if os.getsid(child.pid) == proc.pid and child.status() != psutil.STATUS_ZOMBIE:
                members.append(child)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return members


def _assert_server_listener(proc, port):
    """Reject a port stolen between the initial bind check and server startup."""
    listeners = [
        conn for conn in psutil.net_connections(kind="tcp")
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port
    ]
    owned = {child.pid for child in _server_processes(proc)}
    if any(conn.pid not in owned for conn in listeners):
        raise E2ETestError(f"port {port} is owned by another process; refusing request")
    return bool(listeners)


def _kill_server(proc):
    """Wait for all session members, escalating TERM to KILL when needed."""
    for sig, grace in ((signal.SIGTERM, _TERM_GRACE), (signal.SIGKILL, _KILL_GRACE)):
        deadline, signalled = time.monotonic() + grace, set()
        while True:
            members = _server_processes(proc)
            proc.poll()
            if not members:
                proc.wait(timeout=5)
                return
            for child in members:
                if child not in signalled:
                    try:
                        # psutil checks process identity before signalling reused PIDs.
                        child.send_signal(sig)
                    except psutil.NoSuchProcess:
                        pass
                    signalled.add(child)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
    raise E2ETestError(
        f"server cleanup failed; surviving PIDs: {[p.pid for p in _server_processes(proc)]}"
    )


@contextlib.contextmanager
def _interrupt_cleanup():
    """Turn harness SIGTERM into unwinding so its finally blocks can clean up."""
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt("test interrupted by SIGTERM")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


# Each request includes the previous answers: four turns of one conversation.
_TURNS = (
    "What is the capital of France? Answer briefly.",
    "Which river flows through that city? Answer briefly.",
    "Name one well-known museum in that city. Answer briefly.",
    "Summarize the city, river and museum we discussed in one sentence.",
)
_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_get_status(url: str, timeout: float):
    try:
        with _HTTP.open(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def _wait_ready(case: E2ECase, proc: subprocess.Popen, port: int,
                timeout: int, log_path: Path):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    print(f"[{case.name}] waiting for server on :{port} "
          f"(timeout={timeout}s, log={log_path})", flush=True)
    while time.monotonic() < deadline:
        ret = proc.poll()
        if ret is not None:
            _fail(case, "startup",
                  f"vllm serve exited early with code {ret}", log_path)
        if _assert_server_listener(proc, port) and _http_get_status(url, timeout=5) == 200:
            print(f"[{case.name}] server is ready", flush=True)
            return
        time.sleep(1)
    _fail(case, "startup", f"server not ready within {timeout}s", log_path)


def _assert_fa2_selected(case: E2ECase, log_path: Path):
    """Fail unless the server actually resolved to FA2.

    PPU defaults to FA3, so without this a mis-specified version flag would
    still serve requests happily and the case would prove nothing about FA2.
    """
    try:
        with open(log_path, "r", errors="replace") as f:
            log = f.read()
    except OSError as e:
        _fail(case, "verify-fa2", f"cannot read log: {e}", log_path)
    if _FA2_LOG_MARKER not in log:
        _fail(case, "verify-fa2",
              f"server log never reported {_FA2_LOG_MARKER!r}; "
              f"attention did not run on FA2", log_path)
    print(f"[{case.name}] confirmed: {_FA2_LOG_MARKER}", flush=True)


def _send_completion(case: E2ECase, port: int, log_path: Path, prompt: str) -> str:
    payload = {
        "model": case.served_name,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "stop": ["\nUser:"],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with _HTTP.open(req, timeout=_REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        _fail(case, "request", f"POST /v1/completions failed: {e!r}", log_path)

    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    text = choice.get("text") if isinstance(choice, dict) else None
    usage = body.get("usage") if isinstance(body, dict) else None
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if (not isinstance(text, str) or not text.strip()
            or choice.get("finish_reason") not in ("stop", "length")
            or type(tokens) is not int or not 0 < tokens <= 64):
        _fail(case, "validate", f"invalid completion or generation status: {body!r}", log_path)
    print(f"[{case.name}] response ({tokens} tokens): {text!r}", flush=True)
    return text


def run_case(case: E2ECase, port: int, startup_timeout: int):
    """Boot the server, run four turns, verify FA2 was used, tear down.

    Raises E2ETestError on failure.
    """
    _assert_fa2_wheel_is_local_build()

    if not os.path.isdir(case.model_path):
        raise E2ETestError(
            f"[E2E FAIL] case={case.name} stage=preflight\n"
            f"reason: model path does not exist: {case.model_path}\n"
            f"set the FA2_E2E_*_PATH env var to override."
        )

    port = _select_port(port)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"vllm_e2e_fa2_{case.name}_{os.getpid()}_{time.time_ns()}.log"
    cmd = _build_cmd(case, port)
    print(f"[{case.name}] covers: {case.covers}", flush=True)
    print(f"[{case.name}] launching: {' '.join(cmd)}", flush=True)
    if case.env:
        print(f"[{case.name}] env: {case.env}", flush=True)

    with _interrupt_cleanup(), open(log_path, "w") as log_f:
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, **case.env},
            )
            _wait_ready(case, proc, port, startup_timeout, log_path)
            _assert_fa2_selected(case, log_path)
            history = ""
            for turn, question in enumerate(_TURNS, 1):
                if proc.poll() is not None or not _assert_server_listener(proc, port):
                    _fail(case, "request", "server exited or lost its listener", log_path)
                prompt = history + f"User: {question}\nAssistant:"
                print(f"[{case.name}] turn {turn}/{len(_TURNS)} prompt: {prompt!r}", flush=True)
                answer = _send_completion(case, port, log_path, prompt)
                history = prompt + f" {answer}\n"
                if proc.poll() is not None:
                    _fail(case, "request", "server exited during generation", log_path)
        finally:
            # A repeated Ctrl-C/SIGTERM must not interrupt worker cleanup.
            previous_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
            )
            try:
                if proc is not None:
                    _kill_server(proc)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    print(f"[{case.name}] PASS", flush=True)


# ---------------------------------------------------------------------------
# pytest entry
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.mark.parametrize(
        "case_name", [pytest.param(n, id=n) for n in _CASES]
    )
    def test_vllm_e2e_fa2(case_name):
        port = int(os.environ.get("FA2_E2E_PORT", "8997"))
        run_case(_CASES[case_name], port=port, startup_timeout=_STARTUP_TIMEOUT)

except ImportError:  # pragma: no cover - script mode without pytest
    pass


# ---------------------------------------------------------------------------
# script entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*_CASES, "all"], default="all",
                        help="which case to run (default: all, sequentially)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FA2_E2E_PORT", "8997")),
    )
    parser.add_argument("--timeout", type=int, default=_STARTUP_TIMEOUT,
                        help="server startup timeout in seconds")
    args = parser.parse_args()

    names = list(_CASES) if args.case == "all" else [args.case]
    failures = []
    for name in names:
        case = _CASES[name]
        try:
            run_case(case, port=args.port, startup_timeout=args.timeout)
        except E2ETestError as e:
            print(str(e), file=sys.stderr, flush=True)
            failures.append(name)
        except Exception as e:  # unexpected harness error
            print(f"[E2E FAIL] case={name} unexpected error: {e!r}",
                  file=sys.stderr, flush=True)
            failures.append(name)
        # Give the port a moment to be released before the next case.
        time.sleep(10)

    if failures:
        print(f"\n[E2E] FAILED cases: {failures}", file=sys.stderr)
        return 1
    print("\n[E2E] all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

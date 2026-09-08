"""End-to-end FA3 interface tests driven by a live SGLang server.

Unlike the kernel-level tests in this repository, these cases launch
``sglang serve`` with a real model that routes attention to FA3, send a
four-turn conversation with accumulated context, and verify each response.

On PPU, SGLang's ``fa3`` backend is a thin wrapper around this repo's hopper
``flash_attn_interface`` module (see
``sglang/srt/hardware_backend/ppu/attention/flash_attention.py``, which is
injected into ``flashattention_backend`` by ``ppu_fa3_hooks``), so every case
below exercises the FA3 interfaces end to end.

Covered FA3 entry points (on PPU):

- ``flash_attn_with_kvcache``  — every case: paged-KV extend (prefill) and
  decode. Qwen3.6-35B-A3B / Qwen3.8-27B are hybrid models: only their
  ``full_attention`` layers route to FA3 (10 of 40 and 16 of 64 layers
  respectively), at head_dim 256. gemma-3-1b adds a non-trivial
  ``window_size`` (sliding window).
- ``flash_attn_varlen_func``   — DeepSeek-V2-Lite: ragged MLA prefill with
  unequal qk/v head dims (192 vs 128)
- ``get_scheduler_metadata``   — every case: computed per forward batch before
  decode, including inside CUDA graph replay

Primary cases are the Qwen3.6+ ones (``qwen3.6-35b-a3b``, ``qwen3.8-27b``); the
remaining cases are supplementary because they reach FA3 features no Qwen3.6
model exercises (sliding window, MLA head dims).

Every case fits 2 devices of 96 GB (gemma-3-1b needs only one).

Usage:

    # Run all cases with pytest (each case boots its own server):
    python -m pytest hopper/test_sglang_e2e_fa3.py -v -s

    # Or run directly as a script (exit code 0 = pass, non-zero = fail):
    python hopper/test_sglang_e2e_fa3.py                     # all cases
    python hopper/test_sglang_e2e_fa3.py --case qwen3.6-35b-a3b   # single case

Environment overrides:

    FA3_E2E_QWEN36_35B_PATH     model path for the qwen3.6-35b-a3b case
    FA3_E2E_QWEN38_27B_PATH     model path for the qwen3.8-27b case
    FA3_E2E_DSV2LITE_PATH       model path for the deepseek-v2-lite case
    FA3_E2E_GEMMA3_PATH         model path for the gemma-3-1b case
    FA3_E2E_PORT                server port (default 8998; 0 selects a free port)
    FA3_E2E_TIMEOUT             server startup timeout in seconds (default 1800)
    FA3_E2E_LOG_DIR             directory for server logs (default hopper/logs)

Cases may also carry case-specific env vars for the server process; they are
merged on top of the ambient environment and take precedence.

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
_LOG_DIR = Path(os.environ.get("FA3_E2E_LOG_DIR", _TEST_DIR / "logs"))

_DEFAULT_QWEN36_35B_PATH = "/ppusw/datasets/checkpoints/LLM/qwen/v3.6/Qwen3.6-35B-A3B"
_DEFAULT_QWEN38_27B_PATH = "/ppusw/datasets/checkpoints/LLM/Qwen/v3.8/Qwen3.8-27B"
_DEFAULT_DSV2LITE_PATH = "/ppusw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite"
_DEFAULT_GEMMA3_PATH = "/ppusw/datasets/checkpoints/LLM/gemma/v1.0/gemma-3-1b-it"

_STARTUP_TIMEOUT = int(os.environ.get("FA3_E2E_TIMEOUT", "1800"))
_REQUEST_TIMEOUT = 120
_LOG_TAIL_LINES = 100
_TERM_GRACE = 30
_KILL_GRACE = 15


@dataclasses.dataclass(frozen=True)
class E2ECase:
    """One server-launch test case."""

    name: str
    model_path: str
    # Extra CLI args appended after the common ones.
    extra_args: tuple
    # Human-readable note on which FA3 interfaces this case exercises.
    covers: str = ""
    # Common launch args whose value differs between cases.
    tp_size: int = 2
    mem_fraction_static: str = "0.8"
    # Speculative draft model; preflight-checked and appended when set.
    draft_model_path: str = ""
    # Extra env vars for the server process, merged over os.environ
    # (case values take precedence).
    env: dict = dataclasses.field(default_factory=dict)


_CASES = {
    "qwen3.6-35b-a3b": E2ECase(
        name="qwen3.6-35b-a3b",
        model_path=os.environ.get(
            "FA3_E2E_QWEN36_35B_PATH", _DEFAULT_QWEN36_35B_PATH
        ),
        extra_args=(
            "--served-model-name",
            "Qwen3.6-35B-A3B",
            "--attention-backend",
            "fa3",
            "--cuda-graph-max-bs-decode",
            "8",
        ),
        covers=(
            "flash_attn_with_kvcache (paged extend + decode on the 10 "
            "full_attention layers of a 35B hybrid MoE model, head_dim 256), "
            "get_scheduler_metadata"
        ),
    ),
    "qwen3.8-27b": E2ECase(
        name="qwen3.8-27b",
        model_path=os.environ.get(
            "FA3_E2E_QWEN38_27B_PATH", _DEFAULT_QWEN38_27B_PATH
        ),
        extra_args=(
            "--served-model-name",
            "Qwen3.8-27B",
            "--attention-backend",
            "fa3",
            "--cuda-graph-max-bs-decode",
            "8",
        ),
        covers=(
            "flash_attn_with_kvcache (paged extend + decode on the 16 "
            "full_attention layers of a 27B hybrid model, head_dim 256), "
            "get_scheduler_metadata"
        ),
    ),
    "deepseek-v2-lite": E2ECase(
        name="deepseek-v2-lite",
        model_path=os.environ.get("FA3_E2E_DSV2LITE_PATH", _DEFAULT_DSV2LITE_PATH),
        extra_args=(
            "--served-model-name",
            "DeepSeek-V2-Lite",
            # Same prefill/decode split as the Kimi-K2.5 production config: FA3
            # owns MLA prefill, FlashMLA owns MLA decode. Routing decode to FA3
            # too would hit the `qv` absorb path, which the PPU FA3 kernel
            # rejects with "q_v is only supported for Hopper GPUs".
            "--prefill-attention-backend",
            "fa3",
            "--decode-attention-backend",
            "flashmla",
            "--cuda-graph-max-bs-decode",
            "8",
            # Prefill CUDA graph capture drives MLA extend through the paged
            # `qv` absorb path, which the PPU FA3 kernel rejects; runtime
            # prefill uses ragged flash_attn_varlen_func and works fine.
            "--cuda-graph-backend-prefill",
            "disabled",
        ),
        covers=(
            "flash_attn_varlen_func (ragged MLA prefill, qk head_dim 192 vs "
            "v head_dim 128)"
        ),
    ),
    "gemma-3-1b": E2ECase(
        name="gemma-3-1b",
        model_path=os.environ.get("FA3_E2E_GEMMA3_PATH", _DEFAULT_GEMMA3_PATH),
        extra_args=(
            "--served-model-name",
            "gemma-3-1b-it",
            "--attention-backend",
            "fa3",
            "--cuda-graph-max-bs-decode",
            "8",
            # gemma3_causal.forward calls einops.rearrange on `positions`,
            # which dynamo cannot trace with fake tensors, so capturing the
            # prefill CUDA graph fails. Decode CUDA graph (the one that
            # matters for FA3 metadata + replay) stays enabled.
            "--cuda-graph-backend-prefill",
            "disabled",
        ),
        covers=(
            "flash_attn_with_kvcache with window_size != (-1, -1) "
            "(sliding-window local attention, head_dim 256)"
        ),
        # 1B text-only model; a single device is plenty.
        tp_size=1,
    ),
}


class E2ETestError(RuntimeError):
    """Raised on any e2e failure; carries the full diagnostic message."""


def _build_cmd(case: E2ECase, port: int) -> list:
    cmd = [
        "sglang",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        case.model_path,
        "--tp-size",
        str(case.tp_size),
        "--mem-fraction-static",
        case.mem_fraction_static,
        "--trust-remote-code",
        *case.extra_args,
    ]
    if case.draft_model_path:
        cmd += ["--speculative-draft-model-path", case.draft_model_path]
    return cmd


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
                  f"sglang serve exited early with code {ret}", log_path)
        if _assert_server_listener(proc, port) and _http_get_status(url, timeout=5) == 200:
            print(f"[{case.name}] server is ready", flush=True)
            return
        time.sleep(1)
    _fail(case, "startup", f"server not ready within {timeout}s", log_path)


def _send_generate(case: E2ECase, port: int, log_path: Path, prompt: str) -> str:
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": 64, "temperature": 0, "stop": ["\nUser:"],
        },
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with _HTTP.open(req, timeout=_REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        _fail(case, "request", f"POST /generate failed: {e!r}", log_path)

    text = body.get("text") if isinstance(body, dict) else None
    meta = body.get("meta_info") if isinstance(body, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    reason = meta.get("finish_reason")
    tokens = meta.get("completion_tokens")
    if (not isinstance(text, str) or not text.strip()
            or not isinstance(reason, dict) or reason.get("type") not in ("stop", "length")
            or type(tokens) is not int or not 0 < tokens <= 64):
        _fail(case, "validate", f"invalid generation or finish status: {body!r}", log_path)
    print(f"[{case.name}] response ({tokens} tokens): {text!r}", flush=True)
    return text


def run_case(case: E2ECase, port: int, startup_timeout: int):
    """Boot the server, run four turns, verify, tear down.

    Raises E2ETestError on failure.
    """
    for kind, path in (("model", case.model_path),
                       ("draft model", case.draft_model_path)):
        if path and not os.path.isdir(path):
            raise E2ETestError(
                f"[E2E FAIL] case={case.name} stage=preflight\n"
                f"reason: {kind} path does not exist: {path}\n"
                f"set the FA3_E2E_*_PATH env var to override."
            )

    port = _select_port(port)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"sglang_e2e_{case.name}_{os.getpid()}_{time.time_ns()}.log"
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
            history = ""
            for turn, question in enumerate(_TURNS, 1):
                if proc.poll() is not None or not _assert_server_listener(proc, port):
                    _fail(case, "request", "server exited or lost its listener", log_path)
                prompt = history + f"User: {question}\nAssistant:"
                print(f"[{case.name}] turn {turn}/{len(_TURNS)} prompt: {prompt!r}", flush=True)
                answer = _send_generate(case, port, log_path, prompt)
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
    def test_sglang_e2e_fa3(case_name):
        port = int(os.environ.get("FA3_E2E_PORT", "8998"))
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
        default=int(os.environ.get("FA3_E2E_PORT", "8998")),
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
        time.sleep(5)

    if failures:
        print(f"\n[E2E] FAILED cases: {failures}", file=sys.stderr)
        return 1
    print("\n[E2E] all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

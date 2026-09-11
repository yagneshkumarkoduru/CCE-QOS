"""Run the three-way local vLLM comparison on a single, matched machine.

Configurations are deliberately separated so attribution remains honest:

1. ``fifo_no_prefix_cache``: stock vLLM with FIFO client dispatch.
2. ``fifo_prefix_cache``: stock vLLM FIFO with automatic prefix caching.
3. ``cce_prefix_bounded``: the same vLLM prefix cache plus the CCE-QOS
   bounded prefix-local dispatcher. Queueing is included in latency metrics.

This script is intended for Linux/WSL inside the isolated vLLM environment.
It is not a benchmark reproduction of the vLLM SOSP A100 paper result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .benchmark_protocol import load_trace, trace_sha256
    from .replay import run_replay, write_result
except ImportError:  # Direct script execution from this directory.
    from benchmark_protocol import load_trace, trace_sha256
    from replay import run_replay, write_result


@dataclass(frozen=True)
class RunConfiguration:
    name: str
    enable_prefix_caching: bool
    dispatch_policy: str


CONFIGURATIONS = (
    RunConfiguration("fifo_no_prefix_cache", False, "fifo"),
    RunConfiguration("fifo_prefix_cache", True, "fifo"),
    RunConfiguration("cce_prefix_bounded", True, "prefix_bounded"),
)


def _command_output(command: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _find_pip_cuda_toolkit() -> Optional[Path]:
    """Locate CUDA tools installed beside the isolated Python environment."""
    for site_directory in site.getsitepackages():
        for nvcc_path in sorted(Path(site_directory).glob("nvidia/cu*/bin/nvcc"), reverse=True):
            toolkit_home = nvcc_path.parent.parent
            if (toolkit_home / "include").is_dir() and (toolkit_home / "lib").is_dir():
                return toolkit_home
    return None


def _ensure_cuda_linker_compatibility(toolkit_home: Path) -> Optional[Path]:
    """Provide an unversioned libcudart linker name when pip supplies only an ABI name."""
    library_dir = toolkit_home / "lib"
    if (library_dir / "libcudart.so").is_file():
        return None

    runtime_libraries = sorted(library_dir.glob("libcudart.so.*"))
    if not runtime_libraries:
        return None

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    linker_dir = cache_home / "cce-qos-vllm" / "cuda-linker"
    linker_dir.mkdir(parents=True, exist_ok=True)
    linker_name = linker_dir / "libcudart.so"
    runtime_library = runtime_libraries[-1].resolve()

    if linker_name.exists() or linker_name.is_symlink():
        if linker_name.is_symlink() and linker_name.resolve() == runtime_library:
            return linker_dir
        raise RuntimeError("unexpected libcudart linker entry: %s" % linker_name)

    linker_name.symlink_to(runtime_library)
    return linker_dir


def _server_environment() -> tuple[Dict[str, str], Dict[str, Any]]:
    """Build the child-only environment needed by vLLM and FlashInfer on WSL."""
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("PYTHONHASHSEED", "0")

    path_prefixes = [str(Path(sys.executable).parent)]
    library_prefixes: List[str] = []
    toolkit_home: Optional[Path] = None
    if not environment.get("CUDA_HOME"):
        toolkit_home = _find_pip_cuda_toolkit()
        if toolkit_home is not None:
            toolkit_bin = toolkit_home / "bin"
            toolkit_library = toolkit_home / "lib"
            environment["CUDA_HOME"] = str(toolkit_home)
            path_prefixes.insert(0, str(toolkit_bin))

            linker_directory = _ensure_cuda_linker_compatibility(toolkit_home)
            if linker_directory is not None:
                library_prefixes.append(str(linker_directory))
            library_prefixes.append(str(toolkit_library))

            # WSL exposes the driver library outside the pip CUDA toolkit.
            wsl_driver_library = Path("/usr/lib/wsl/lib")
            if wsl_driver_library.is_dir():
                library_prefixes.append(str(wsl_driver_library))

            linker_flags = ["-L%s" % directory for directory in library_prefixes]
            existing_flags = environment.get("FLASHINFER_EXTRA_LDFLAGS")
            if existing_flags:
                linker_flags.append(existing_flags)
            environment["FLASHINFER_EXTRA_LDFLAGS"] = " ".join(linker_flags)

    existing_path = environment.get("PATH")
    environment["PATH"] = os.pathsep.join(path_prefixes + ([existing_path] if existing_path else []))
    if library_prefixes:
        existing_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            library_prefixes + ([existing_library_path] if existing_library_path else [])
        )

    # vLLM 0.29.0 Model Runner V2 requires UVA, while vLLM disables pinned
    # memory on WSL2 by default. vLLM issue #54652 documents the resulting
    # startup failure and the upstream-supported V1 fallback.
    if "microsoft" in platform.release().lower():
        environment.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    return environment, {
        "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONHASHSEED": environment.get("PYTHONHASHSEED"),
        "VLLM_USE_V2_MODEL_RUNNER": environment.get("VLLM_USE_V2_MODEL_RUNNER"),
        "CUDA_HOME": environment.get("CUDA_HOME"),
        "pip_cuda_toolkit": str(toolkit_home) if toolkit_home is not None else None,
        "PATH_prepend": path_prefixes,
        "LD_LIBRARY_PATH_prepend": library_prefixes,
        "FLASHINFER_EXTRA_LDFLAGS": environment.get("FLASHINFER_EXTRA_LDFLAGS"),
    }


def _environment_metadata() -> Dict[str, Any]:
    gpu = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "vllm_version": _command_output([sys.executable, "-c", "import vllm; print(vllm.__version__)"]),
        "torch_version": _command_output([sys.executable, "-c", "import torch; print(torch.__version__)"]),
        "gpu": gpu,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command_output(["git", "status", "--porcelain"])),
    }


def _wait_for_server(endpoint: str, process: subprocess.Popen[Any], timeout_s: float, log_path: Path) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required in the vLLM benchmark environment") from exc

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server stopped before readiness; inspect %s" % log_path)
        try:
            response = httpx.get(endpoint.rstrip("/") + "/v1/models", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError("vLLM server did not become ready within %.1f seconds; inspect %s" % (timeout_s, log_path))


def _start_server(
    model: str,
    endpoint: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    log_path: Path,
) -> tuple[subprocess.Popen[Any], List[str], Any, Dict[str, Any]]:
    host_port = endpoint.removeprefix("http://").removeprefix("https://").split(":")
    if len(host_port) != 2 or not host_port[1].isdigit():
        raise ValueError("endpoint must be http://host:port")
    environment_binary = Path(sys.executable).with_name("vllm")
    vllm_binary = str(environment_binary) if environment_binary.is_file() else shutil.which("vllm")
    if vllm_binary is None:
        raise RuntimeError("vllm executable not found; activate the isolated benchmark environment")

    command = [
        vllm_binary,
        "serve",
        model,
        "--host",
        host_port[0],
        "--port",
        host_port[1],
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--seed",
        str(seed),
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    else:
        command.append("--no-enable-prefix-caching")

    environment, server_environment = _server_environment()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    return process, command, handle, server_environment


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30.0)


def _warm_server(endpoint: str, model: str, request_count: int, request_timeout_s: float) -> None:
    """Warm model loading/graphs without adding a trace prefix to the cache."""
    if request_count <= 0:
        return
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required in the vLLM benchmark environment") from exc
    payload = {
        "model": model,
        "prompt": "Warmup request for benchmark initialization only.",
        "max_tokens": 4,
        "temperature": 0.0,
        "seed": 0,
    }
    with httpx.Client(timeout=request_timeout_s) as client:
        for _ in range(request_count):
            response = client.post(endpoint.rstrip("/") + "/v1/completions", json=payload)
            response.raise_for_status()


def run_configuration(
    configuration: RunConfiguration,
    trace_path: Path,
    result_dir: Path,
    model: str,
    endpoint: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_inflight: int,
    max_reorder_wait_ms: int,
    request_timeout_s: float,
    server_timeout_s: float,
    warmup_requests: int,
    enforce_eager: bool,
    seed: int,
) -> Path:
    trace = load_trace(trace_path)
    log_path = result_dir / (configuration.name + ".server.log")
    process, command, handle, server_environment = _start_server(
        model=model,
        endpoint=endpoint,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        seed=seed,
        enable_prefix_caching=configuration.enable_prefix_caching,
        enforce_eager=enforce_eager,
        log_path=log_path,
    )
    try:
        _wait_for_server(endpoint, process, server_timeout_s, log_path)
        _warm_server(endpoint, model, warmup_requests, request_timeout_s)
        result = asyncio.run(
            run_replay(
                endpoint=endpoint,
                model=model,
                requests=trace,
                policy=configuration.dispatch_policy,
                max_inflight=max_inflight,
                max_reorder_wait_ms=max_reorder_wait_ms,
                request_timeout_s=request_timeout_s,
                seed=seed,
            )
        )
        result["run_metadata"] = {
            "configuration": configuration.name,
            "automatic_prefix_caching": configuration.enable_prefix_caching,
            "dispatch_policy": configuration.dispatch_policy,
            "model": model,
            "endpoint": endpoint,
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": trace_sha256(trace),
            "server_command": command,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_inflight": max_inflight,
            "max_reorder_wait_ms": max_reorder_wait_ms,
            "warmup_requests": warmup_requests,
            "enforce_eager": enforce_eager,
            "seed": seed,
            "server_log": str(log_path.resolve()),
            "server_environment": server_environment,
            "environment": _environment_metadata(),
        }
    finally:
        _stop_server(process)
        handle.close()

    output_path = result_dir / (configuration.name + ".json")
    write_result(output_path, result)
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run matched stock-vLLM and CCE-QOS dispatch comparisons.")
    parser.add_argument("--trace", type=Path, required=True, help="Canonical JSONL trace from make_prefix_trace.py.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / time.strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-inflight", type=int, default=16)
    parser.add_argument("--max-reorder-wait-ms", type=int, default=100)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--server-timeout-s", type=float, default=600.0)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs only for debugging; default uses stock vLLM optimization.",
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--only",
        choices=[configuration.name for configuration in CONFIGURATIONS],
        default=None,
        help="Run one configuration instead of the full three-way attribution set.",
    )
    args = parser.parse_args(argv)

    if not args.trace.is_file():
        parser.error("trace does not exist: %s" % args.trace)
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_model_len <= 0 or args.max_num_seqs <= 0 or args.max_inflight <= 0:
        parser.error("sequence and inflight limits must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")

    configurations = [
        configuration for configuration in CONFIGURATIONS if args.only in (None, configuration.name)
    ]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []
    for configuration in configurations:
        print("running %s" % configuration.name, flush=True)
        output = run_configuration(
            configuration=configuration,
            trace_path=args.trace,
            result_dir=args.results_dir,
            model=args.model,
            endpoint=args.endpoint,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_inflight=args.max_inflight,
            max_reorder_wait_ms=args.max_reorder_wait_ms,
            request_timeout_s=args.request_timeout_s,
            server_timeout_s=args.server_timeout_s,
            warmup_requests=args.warmup_requests,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
        )
        outputs.append(str(output))
    print(json.dumps({"result_files": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

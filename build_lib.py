#!/usr/bin/env python3
"""Build libflex_flash_attention.so — the pure C++ library consumed by the
pytorch SDPA integration (cmake/flex_flash_attention.cmake).

Deliberately torch-import-free: during a torch source build there is no
importable torch yet, so this script drives nvcc/g++ directly.  Torch
include directories and ABI macros are injected by the caller via
--include / --define; kernel-side flags below are a frozen snapshot of
setup.py's USE_PPU branch (keep the two in sync when editing either).

Usage (invoked by cmake, but runnable standalone):
    python build_lib.py --out /path/libflex_flash_attention.so \
        --obj-dir /path/objs \
        --include <torch-src>/aten/src --include <torch-build>/aten/src \
        --define _GLIBCXX_USE_CXX11_ABI=1 \
        [--nvcc /path/nvcc] [--cxx /path/g++] [-j N]

Kernel-side headers (the private flex_flash_attention/hopper/ copy, actlize
cutlass3, PPU_SDK targets) are resolved from the repo layout; point
ACTLIZE_DIR at the actlize repo if the csrc/actlize symlink is absent.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # repo root (module flattened here)
REPO_DIR = THIS_DIR                                 # module is the repo root now
HOPPER_DIR = REPO_DIR / "hopper"                    # module-private hopper/ copy
# Load sources.py by path: the flex_flash_attention package __init__ imports
# torch, which is unavailable during a torch source build.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_flex_flash_attention_sources", THIS_DIR / "sources.py")
_sources_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sources_mod)
flex_flash_sources = _sources_mod.flex_flash_sources

# ---- kernel flags: snapshot of hopper/setup.py (USE_PPU branch) ---------
NVCC_FLAGS = [
    "-O3", "-std=c++17",
    "--ftemplate-backtrace-limit=0",
    "--use_fast_math",
    "--resource-usage",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    # USE_PPU branch
    "-mllvm", "-ppu-max-vreg-count=256",
    "-mllvm", "-ppu-sink-matrix-addr=true",
    "-mllvm", "-ppu-max-alloca-byte-size=320",
    "-mllvm", "-ppu-sink-async-addr=true",
    "-mllvm", "-ppu-sink-load-addr=true",
    "-mllvm", "-ppu-sink-store-addr=true",
    "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
    "-mllvm", "-ppu-volatile-yield=false",
    "-mllvm", "-sort-copy-before-coalesce=true",
    "-DUSE_PPU", "-DUSE_AIU=1",
    "-Xfatbin", "--compress-all",
    "-D__NV_SILENCE_DEPRECATION_BEGIN=",
    "-D__NV_SILENCE_DEPRECATION_END=",
    # feature_args: USE_PPU defaults DISABLE_SM90 (setup.py L86) and always
    # disables SM86 — the actlize cutlass has no SM90 builder headers, so
    # the guarded sm90_common.inl includes must stay out.
    "-DFLASHATTENTION_DISABLE_SM90",
    "-DFLASHATTENTION_DISABLE_SM86",
]

# FLEX_FLASH_ATTENTION_LIBRARY_BUILD skips the wheel-only Python.h/PyInit bits in
# flex_flash_attention_api.cpp so the library build needs no Python headers.
CXX_FLAGS = ["-O3", "-std=c++17", "-DNDEBUG", "-DUSE_PPU", "-DUSE_AIU=1",
             "-DFLASHATTENTION_DISABLE_SM90",
             "-DFLASHATTENTION_DISABLE_SM86",
             "-DFLEX_FLASH_ATTENTION_LIBRARY_BUILD"]
# All flex flash attention TUs are native sm89 (or arch-independent logic);
# a single gencode keeps the build simple and matches the PPU target.
GENCODE = ["-gencode", "arch=compute_89,code=sm_89"]


def kernel_include_dirs():
    # The kernel headers live in the module-private flex_flash_attention/hopper/
    # copy and are reached via relative quote-includes; hopper/ itself is
    # the pristine upstream baseline and stays OUT of the include chain.
    # The repo root goes LAST so "flex_flash_attention/csrc/..." includes
    # resolve.
    dirs = [str(THIS_DIR / "include"), str(REPO_DIR)]
    # Repo-local cutlass3 lives in the actlize submodule (see setup.py).
    # Resolution order: ACTLIZE_DIR env (absolute path to the actlize repo),
    # then the repo-local csrc/actlize symlink; both must expose include/.
    candidates = []
    env_actlize = os.environ.get("ACTLIZE_DIR", "")
    if env_actlize:
        candidates.append(Path(env_actlize))
    candidates.append(REPO_DIR / "csrc" / "actlize")
    for cand in candidates:
        if (cand / "include").is_dir():
            dirs.append(str(cand / "include"))
            break
    else:
        raise RuntimeError(
            "actlize (cutlass3 headers) not found: set ACTLIZE_DIR to the "
            "actlize repo root or provide csrc/actlize symlink")
    ppu_sdk = os.environ.get("PPU_SDK", "")
    if ppu_sdk:
        tgt = os.path.join(ppu_sdk, "targets", "x86_64-linux", "include")
        if os.path.isdir(tgt):
            dirs.insert(0, tgt)
    return dirs


def sources():
    # Same TU set as the wheel, minus flash_api.cpp (dense FA3 entry, not
    # needed by SDPA) and plus the SDPA glue TU.  After the flatten every
    # manifest entry is repo-root-relative (csrc/..., instantiations/...),
    # so they all resolve straight from REPO_DIR (== the repo root).
    srcs = [s for s in flex_flash_sources()
            if not s.startswith("flash_api")]
    srcs = [REPO_DIR / s for s in srcs]
    srcs.append(THIS_DIR / "csrc" / "sdpa_glue.cpp")
    return srcs


def header_stamp():
    # All TUs share the csrc/include/hopper headers; a header edit must
    # rebuild every TU even though the .cu mtime is unchanged.
    newest = 0.0
    for d in (THIS_DIR / "csrc", THIS_DIR / "include", THIS_DIR / "hopper"):
        for f in d.iterdir():
            if f.is_file():
                newest = max(newest, os.path.getmtime(f))
    return newest


_HDR_STAMP = None


def compile_one(args):
    global _HDR_STAMP
    src, obj, cmd = args
    if _HDR_STAMP is None:
        _HDR_STAMP = header_stamp()
    if os.path.exists(obj) and os.path.getmtime(obj) >= max(
            os.path.getmtime(src), _HDR_STAMP):
        return src, "cached"
    os.makedirs(os.path.dirname(obj), exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return src, f"FAILED:\n{r.stdout}\n{r.stderr}"
    return src, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--obj-dir", required=True)
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--define", action="append", default=[])
    # Explicit nvcc (cmake passes CMAKE_CUDA_COMPILER so kernel TUs use the
    # same compiler the main build uses).
    ap.add_argument("--nvcc", default=None)
    # Explicit host compiler for .cpp TUs and the final link (cmake passes
    # CMAKE_CXX_COMPILER to keep ABI/stdlib in sync with libtorch).
    ap.add_argument("--cxx", default=None)
    ap.add_argument("-j", type=int, default=min(16, os.cpu_count() or 8))
    args = ap.parse_args()

    nvcc = args.nvcc or os.environ.get("PYTORCH_NVCC") or "nvcc"
    cxx = args.cxx or os.environ.get("CXX", "g++")
    os.environ.setdefault("HGGC_ENABLE_COMPRESS", "1")

    inc_flags = [f"-I{d}" for d in kernel_include_dirs() + args.include]
    def_flags = [f"-D{d}" for d in args.define]

    jobs = []
    for src in sources():
        src = Path(src)
        obj = os.path.join(args.obj_dir, src.stem + ".o")
        if src.suffix == ".cu":
            cmd = [nvcc, *NVCC_FLAGS, *inc_flags, *def_flags,
                   "-Xcompiler", "-fPIC",
                   *GENCODE, "-c", str(src), "-o", obj]
        else:
            cmd = [cxx, *CXX_FLAGS, *inc_flags, *def_flags,
                   "-fPIC", "-c", str(src), "-o", obj]
        jobs.append((str(src), obj, cmd))

    with ProcessPoolExecutor(max_workers=args.j) as ex:
        results = list(ex.map(compile_one, jobs))
    failed = [(s, msg) for s, msg in results if msg.startswith("FAILED")]
    for src, msg in failed:
        print(f"== {src} ==\n{msg}", file=sys.stderr)
    if failed:
        sys.exit(1)
    print(f"compiled {len(jobs)} TUs "
          f"({sum(1 for _, m in results if m == 'cached')} cached)")

    objs = [obj for _, obj, _ in jobs]
    # SONAME so consumers' DT_NEEDED records the bare name (resolved via
    # $ORIGIN runpath next to libtorch_*.so); without it the linker records
    # the link-time path, which breaks at runtime after install.
    link = [cxx, "-shared", "-Wl,-soname,libflex_flash_attention.so", *objs,
            "-o", args.out]
    r = subprocess.run(link, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"link FAILED:\n{r.stdout}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"linked {args.out}")


if __name__ == "__main__":
    main()

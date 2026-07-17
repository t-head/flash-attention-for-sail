# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# CUDA-free PPU build using hgcc.

import sys
import os
import re
import ast
import itertools
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch

PACKAGE_NAME = "flash_attn"
this_dir = os.path.dirname(os.path.abspath(__file__))

SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
FA3_HLLM_BUILD = os.getenv("FA3_HLLM_BUILD", "0") == "1"
FA3_HLLM_USE_ADDR = os.getenv("FA3_HLLM_USE_ADDR", "0") == "1"

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = True  # FP8 not available in fa274 hopper instantiations
DISABLE_VARLEN = os.getenv("FLASH_ATTENTION_DISABLE_VARLEN", "FALSE") == "TRUE"
DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", "TRUE") == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM8x", "FALSE") == "TRUE"
DISABLE_SM90 = True  # SM90 removed from this PPU-only build
ENABLE_VCOLMAJOR = os.getenv("FLASH_ATTENTION_ENABLE_VCOLMAJOR", "FALSE") == "TRUE"


# ============================================================================
# PPU HGCC Build Extension
# ============================================================================

class HGCCBuildExtension(build_ext):
    """Custom build extension that uses hgcc for PPU."""

    def build_extensions(self):
        if not os.environ.get("MAX_JOBS"):
            os.environ["MAX_JOBS"] = "32"

        for ext in self.extensions:
            self._build_extension_hgcc(ext)

    def _build_extension_hgcc(self, ext):
        import ninja  # noqa: F401

        ppu_sdk = os.environ.get("PPU_SDK", "")
        hgcc = os.path.join(ppu_sdk, "bin", "hgcc")
        torch_dir = torch.__path__[0]

        sources = [os.path.join(this_dir, s) for s in ext.sources]
        include_dirs = [os.path.abspath(d) for d in ext.include_dirs]

        output_dir = os.path.join(self.build_temp, "hgcc_objs")
        os.makedirs(output_dir, exist_ok=True)

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        torch_include = os.path.join(torch_dir, "include")
        torch_include_csrc = os.path.join(torch_dir, "include", "torch", "csrc", "api", "include")
        python_include = subprocess.check_output(
            [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
        ).decode().strip()

        actlize_shims = os.path.join(this_dir, "..", "csrc", "actlize", "cmake", "ppu_sdk_shims")
        all_includes = [actlize_shims] + include_dirs + [torch_include, torch_include_csrc, python_include]
        include_flags = [f"-I{d}" for d in all_includes]

        hgcc_flags = [
            "-O3", "-std=c++17",
            "-arch=ppu_10",
            "--forward-unknown-to-host-compiler",
            "-Xcompiler", "-fPIC",
            "-DSWITCH_TO_HGGCRT",
            "-DUSE_CLANG", "-DUSE_HGGC", "-DUSE_PPU", "-DUSE_AIU=1",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-DTORCH_EXTENSION_NAME={ext.name.split('.')[-1]}",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--resource-usage",
            "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
            "-DCUTLASS_ENABLE_GDC_FOR_SM90",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "-DNDEBUG",
            "-mllvm", "-ppu-max-vreg-count=256",
            "-mllvm", "-ppu-patch-fence-ppu=false",
            "-mllvm", "-ppu-fix-uninit=true",
            "-mllvm", "-wno-loop-miss-transform",
        ] + ext.extra_compile_args.get("hgcc_extra", [])

        ppu_sdk_inc = os.path.join(ppu_sdk, "include")
        ppu_targets_inc = os.path.join(ppu_sdk, "targets", "x86_64-linux", "include")
        cxx_flags = [
            "-O3", "-std=c++17", "-fPIC",
            "-DUSE_PPU", "-DUSE_AIU=1",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            f"-DTORCH_EXTENSION_NAME={ext.name.split('.')[-1]}",
            "-I" + ppu_sdk_inc,
            "-I" + ppu_targets_inc,
        ] + ext.extra_compile_args.get("cxx_extra", [])

        max_jobs = int(os.environ.get("MAX_JOBS", "4"))
        ninja_file = os.path.join(output_dir, "build.ninja")
        obj_files = []

        with open(ninja_file, "w") as f:
            f.write("ninja_required_version = 1.3\n\n")

            f.write(f"rule hgcc_compile\n")
            f.write(f"  command = {hgcc} {' '.join(hgcc_flags)} {' '.join(include_flags)} -c $in -o $out\n")
            f.write(f"  description = HGCC $in\n\n")

            cxx_compiler = "c++"
            cxx_include_flags = [f"-I{d}" for d in all_includes]
            f.write(f"rule cxx_compile\n")
            f.write(f"  command = {cxx_compiler} {' '.join(cxx_flags)} {' '.join(cxx_include_flags)} -c $in -o $out\n")
            f.write(f"  description = CXX $in\n\n")

            torch_lib_dir = os.path.join(torch_dir, "lib")
            ppu_lib_dir = os.path.join(ppu_sdk, "lib")
            link_libs = f"-L{torch_lib_dir} -L{ppu_lib_dir} -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda -lhggc_wrapper -lhg_wrapper"
            f.write(f"rule link\n")
            f.write(f"  command = {hgcc} -shared -o $out $in {link_libs}\n")
            f.write(f"  description = LINK $out\n\n")

            for src in sources:
                basename = os.path.splitext(os.path.basename(src))[0]
                obj = os.path.join(output_dir, basename + ".o")
                obj_files.append(obj)

                if src.endswith(".cu"):
                    f.write(f"build {obj}: hgcc_compile {src}\n")
                else:
                    f.write(f"build {obj}: cxx_compile {src}\n")

            f.write(f"\nbuild {ext_path}: link {' '.join(obj_files)}\n")
            f.write(f"\ndefault {ext_path}\n")

        print(f"\n[HGCCBuildExtension] Building {ext.name} with {len(sources)} sources, max_jobs={max_jobs}")
        subprocess.check_call(["ninja", "-f", ninja_file, f"-j{max_jobs}"])
        print(f"[HGCCBuildExtension] Built {ext_path}")


# ============================================================================
# Source file generation
# ============================================================================

ext_modules = []

dir_actlize = this_dir + "/../csrc/actlize"
if not os.path.exists(dir_actlize):
    repo_actlize = os.path.dirname(this_dir) + "/actlize"
    if not os.path.exists(repo_actlize):
        raise RuntimeError(f"actlize does not exist: \"{repo_actlize}\" or \"{dir_actlize}\"")
    else:
        os.symlink(repo_actlize, dir_actlize)

if not SKIP_CUDA_BUILD:
    print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")

    repo_dir = Path(this_dir).parent
    actlize_dir = repo_dir / "csrc" / "actlize"

    feature_args = (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_PAGEDKV"] if DISABLE_PAGEDKV else [])
        + (["-DFLASHATTENTION_DISABLE_SPLIT"] if DISABLE_SPLIT else [])
        + (["-DFLASHATTENTION_DISABLE_APPENDKV"] if DISABLE_APPENDKV else [])
        + (["-DFLASHATTENTION_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DFLASHATTENTION_DISABLE_SOFTCAP"] if DISABLE_SOFTCAP else [])
        + (["-DFLASHATTENTION_DISABLE_PACKGQA"] if DISABLE_PACKGQA else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DFLASHATTENTION_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DFLASHATTENTION_DISABLE_VARLEN"] if DISABLE_VARLEN else [])
        + (["-DFLASHATTENTION_DISABLE_CLUSTER"] if DISABLE_CLUSTER else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if DISABLE_SM8x else [])
        + ["-DFLASHATTENTION_DISABLE_SM86"]
        + (["-DFLASHATTENTION_DISABLE_SM90"] if DISABLE_SM90 else [])
        + (["-DFLASHATTENTION_ENABLE_VCOLMAJOR"] if ENABLE_VCOLMAJOR else [])
        + ["-DUSE_PPU", "-DUSE_AIU=1"]
        + (["-DFA3_HLLM_BUILD"] if FA3_HLLM_BUILD else [])
        + (["-DFA3_HLLM_USE_ADDR"] if FA3_HLLM_USE_ADDR else [])
    )

    DTYPE_FWD_SM80 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else []) + (["e4m3"] if not DISABLE_FP8 else [])
    DTYPE_BWD = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS_BWD = (
        []
        + ([64] if not DISABLE_HDIM64 else [])
        + ([96] if not DISABLE_HDIM96 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([192] if not DISABLE_HDIM192 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    HEAD_DIMENSIONS_FWD_SM80 = HEAD_DIMENSIONS_BWD
    SPLIT = [""] + (["_split"] if not DISABLE_SPLIT else [])
    PAGEDKV = [""] + (["_paged"] if not DISABLE_PAGEDKV else [])
    SOFTCAP = [""] + (["_softcap"] if not DISABLE_SOFTCAP else [])
    SOFTCAP_ALL = [""] if DISABLE_SOFTCAP else ["_softcapall"]
    PACKGQA = [""] + (["_packgqa"] if not DISABLE_PACKGQA else [])

    sources_fwd_sm80 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm80.cu"
                        for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_FWD_SM80, DTYPE_FWD_SM80, SPLIT, PAGEDKV, SOFTCAP_ALL, PACKGQA)
                        if not ((packgqa and (paged or split)) or (dtype == "e4m3" and hdim == 96))]
    sources_bwd_sm80 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm80.cu"
                        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP)]

    if DISABLE_BACKWARD:
        sources_bwd_sm80 = []

    sources = (
        ["flash_api.cpp"]
        + (sources_fwd_sm80 if not DISABLE_SM8x else [])
        + (sources_bwd_sm80 if not DISABLE_SM8x else [])
    )
    if not DISABLE_SPLIT:
        sources += ["flash_fwd_combine.cu"]

    include_dirs = [
        str(Path(this_dir)),
        str(actlize_dir / "include"),
    ]

    os.environ["HGGC_ENABLE_COMPRESS"] = "1"

    ext_modules.append(
        Extension(
            name="flash_attn_3_cuda",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "hgcc_extra": feature_args,
                "cxx_extra": feature_args,
            },
        )
    )


# ============================================================================
# Package setup
# ============================================================================

def get_package_version():
    with open(Path(this_dir) / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        return super().run()


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    py_modules=["flash_attn_interface"],
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention 3: Fast and Memory-Efficient Exact Attention",
    long_description=open("../README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": HGCCBuildExtension},
    python_requires=">=3.9",
    install_requires=["torch", "einops"],
    setup_requires=["packaging", "ninja"],
)

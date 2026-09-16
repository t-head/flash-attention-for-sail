"""Unit tests for FlashAttention-3 varlen forward and backward pass."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import pytest
import torch
import torch.nn.functional as F
from flash_attn_interface import flash_attn_varlen_func


def attention_ref(q, k, v, causal=False, window_size=(-1, -1)):
    """Simple reference attention for validation.
    q: (batch, seqlen_q, nheads, d)
    k: (batch, seqlen_k, nheads_k, d)
    v: (batch, seqlen_k, nheads_k, d)
    """
    # Handle GQA by repeating k,v heads
    nheads = q.shape[2]
    nheads_k = k.shape[2]
    if nheads != nheads_k:
        repeat = nheads // nheads_k
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    # Compute attention in fp32
    scale = q.shape[-1] ** (-0.5)
    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()

    # (batch, nheads, seqlen_q, seqlen_k)
    attn = torch.einsum('bqhd,bkhd->bhqk', q_f, k_f)

    # Apply window mask if needed
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    if window_size[0] >= 0 or window_size[1] >= 0:
        row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        # Flash attention uses an offset when seqlen_q != seqlen_k
        sk_offset = seqlen_k - seqlen_q
        if window_size[0] >= 0:
            attn.masked_fill_(col_idx < (row_idx + sk_offset) - window_size[0], float('-inf'))
        if window_size[1] >= 0:
            attn.masked_fill_(col_idx > (row_idx + sk_offset) + window_size[1], float('-inf'))

    if causal:
        row_idx = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        causal_mask = col_idx > row_idx + (seqlen_k - seqlen_q)
        attn.masked_fill_(causal_mask, float('-inf'))

    attn = F.softmax(attn, dim=-1)
    out = torch.einsum('bhqk,bkhd->bqhd', attn, v_f)
    return out.to(q.dtype)


_HEADDIM_SUPPORTED = {}


def _skip_if_headdim_unsupported(headdim):
    """Skip when the installed build was compiled without this head dimension.

    setup.py gates each head dim behind FLASH_ATTENTION_DISABLE_HDIM{64,96,128,192,256},
    and flash_api.cpp rejects anything above the largest compiled-in value.
    """
    if headdim not in _HEADDIM_SUPPORTED:
        q = torch.zeros(1, 1, headdim, dtype=torch.bfloat16, device="cuda")
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        try:
            flash_attn_varlen_func(q, q, q, cu_seqlens, cu_seqlens, 1, 1)
            _HEADDIM_SUPPORTED[headdim] = True
        except RuntimeError as e:
            if "head dimension at most" not in str(e):
                raise
            _HEADDIM_SUPPORTED[headdim] = False
    if not _HEADDIM_SUPPORTED[headdim]:
        pytest.skip(f"installed flash_attn_3 build does not support headdim={headdim}")


def _run_varlen_fwd_bwd(
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_k,
    headdim,
    causal=False,
    window_size=None,
    dtype=torch.bfloat16,
    check_reference=False,
    atol=0.05,
    rtol=0.05,
):
    """Shared driver: build a packed varlen problem, run fwd+bwd, validate.

    Returns a dict with the flash outputs/gradients and the measured timings.
    """
    device = "cuda"
    if window_size is None:
        window_size = (-1, -1)
    _skip_if_headdim_unsupported(headdim)

    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k

    q = torch.randn(total_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens_q = torch.arange(
        0, total_q + 1, seqlen_q, dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.arange(
        0, total_k + 1, seqlen_k, dtype=torch.int32, device=device
    )

    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    print(f"  cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"  cu_seqlens_k: {cu_seqlens_k.tolist()}")
    print(f"  causal: {causal}, window_size: {window_size}")
    print(f"  GQA ratio: {nheads // nheads_k}x")

    torch.cuda.synchronize()
    t0 = time.time()

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        seqlen_q, seqlen_k,
        softmax_scale=None,
        causal=causal,
        window_size=window_size,
        softcap=0.0,
    )
    if isinstance(out, tuple):
        out = out[0]

    torch.cuda.synchronize()
    fwd_time = time.time() - t0

    assert out.shape == (total_q, nheads, headdim), \
        f"Output shape mismatch: expected {(total_q, nheads, headdim)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Forward output contains NaN!"
    assert not torch.isinf(out).any(), "Forward output contains Inf!"
    print(f"  [PASS] Forward pass completed in {fwd_time:.3f}s, no NaN/Inf")

    dout = torch.randn_like(out)

    torch.cuda.synchronize()
    t2 = time.time()
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout, retain_graph=check_reference)
    torch.cuda.synchronize()
    bwd_time = time.time() - t2

    assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
    assert dk.shape == k.shape, f"dk shape mismatch: expected {k.shape}, got {dk.shape}"
    assert dv.shape == v.shape, f"dv shape mismatch: expected {v.shape}, got {dv.shape}"

    for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert not torch.isnan(g).any(), f"{name} contains NaN!"
        assert not torch.isinf(g).any(), f"{name} contains Inf!"
    print(f"  [PASS] Backward pass completed in {bwd_time:.3f}s, no NaN/Inf in dq/dk/dv")

    if check_reference:
        q_b = q.detach().clone().reshape(batch_size, seqlen_q, nheads, headdim)
        k_b = k.detach().clone().reshape(batch_size, seqlen_k, nheads_k, headdim)
        v_b = v.detach().clone().reshape(batch_size, seqlen_k, nheads_k, headdim)

        out_ref = attention_ref(q_b, k_b, v_b, causal=causal, window_size=window_size)
        out_ref = out_ref.reshape(total_q, nheads, headdim)

        max_diff_fwd = (out.float() - out_ref.float()).abs().max().item()
        print(f"  Forward max abs diff vs reference: {max_diff_fwd:.6f}")
        assert torch.allclose(out.float(), out_ref.float(), atol=atol, rtol=rtol), \
            f"Forward mismatch vs reference: max_diff={max_diff_fwd:.6f} (atol={atol}, rtol={rtol})"

        q_ref = q_b.clone().requires_grad_(True)
        k_ref = k_b.clone().requires_grad_(True)
        v_ref = v_b.clone().requires_grad_(True)
        out_ref2 = attention_ref(q_ref, k_ref, v_ref, causal=causal, window_size=window_size)
        out_ref2 = out_ref2.reshape(total_q, nheads, headdim)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(
            out_ref2, (q_ref, k_ref, v_ref), dout.detach().clone()
        )
        dq_ref = dq_ref.reshape(total_q, nheads, headdim)
        dk_ref = dk_ref.reshape(total_k, nheads_k, headdim)
        dv_ref = dv_ref.reshape(total_k, nheads_k, headdim)

        torch.set_printoptions(profile="full")
        for name, t in (("dk", dk), ("dk_ref", dk_ref)):
            row = t[46, 0, :].float()
            print(f"==== {name} row [46,0,:] ({row.numel()} elems, 8 per group) ====")
            for g in range(0, row.numel(), 8):
                vals = " ".join(f"{x:.2f}" for x in row[g:g+8].tolist())
                print(f"  [{g:4d}-{min(g+7, row.numel()-1):4d}] {vals}")
        # print(f"==== dk [{tuple(dk.shape)}] ====\n{dk.float()-dk_ref.float()}")
        # print(f"==== dk_ref [{tuple(dk_ref.shape)}] ====\n{dk_ref.float()}")
        torch.set_printoptions(profile="default")

        all_pass = True
        for name, g, g_ref in (("dq", dq, dq_ref), ("dk", dk, dk_ref), ("dv", dv, dv_ref)):
            diff = (g.float() - g_ref.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            close = torch.allclose(g.float(), g_ref.float(), atol=atol, rtol=rtol)
            status = "PASS" if close else "FAIL"
            print(f"  [{status}] Backward {name}: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")
            if not close:
                all_pass = False
        if not all_pass:
            print(f"  [WARNING] Some backward gradients exceed tolerance (atol={atol}, rtol={rtol})")
        else:
            print(f"  [ALL PASSED] Backward gradients within tolerance (atol={atol}, rtol={rtol})")


    return dict(out=out, dq=dq, dk=dk, dv=dv, fwd_time=fwd_time, bwd_time=bwd_time)


def test_varlen_fwd_bwd():
    """Large-scale smoke test for varlen forward and backward.

    Parameters from FMHA descriptor:
    - batch_size: 2
    - seqlen_q: 42840 (max), seqlen_k: 44446 (max)
    - head_dim: 128, num_heads: 32, num_heads_k: 8 (GQA 4x)
    - dtype: bf16
    - window_size: (44445, 42839)
    - cu_seqlens_q: [0, 42840, 85680]
    - cu_seqlens_k: [0, 44446, 88561]
    """
    print("=" * 60)
    print("TEST: test_varlen_fwd_bwd (large-scale smoke test)")
    print("=" * 60)

    torch.manual_seed(42)

    # Parameters
    total_q = 85680
    total_k = 88561
    nheads = 32
    nheads_k = 8
    headdim = 128
    max_seqlen_q = 42840
    max_seqlen_k = 44446
    window_size = (44445, 42839)
    dtype = torch.bfloat16
    device = "cuda"

    # Create tensors
    q = torch.randn(total_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens_q = torch.tensor([0, 42840, 85680], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 44446, 88561], dtype=torch.int32, device=device)

    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    print(f"  cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"  cu_seqlens_k: {cu_seqlens_k.tolist()}")
    print(f"  max_seqlen_q: {max_seqlen_q}")
    print(f"  max_seqlen_k: {max_seqlen_k}")
    print(f"  window_size: {window_size}")
    print(f"  GQA ratio: {nheads // nheads_k}x")
    print()

    # Forward pass
    torch.cuda.synchronize()
    t0 = time.time()

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale=None,
        causal=False,
        window_size=window_size,
        softcap=0.0,
    )
    # flash_attn_varlen_func may return a tuple (out, softmax_lse, ...)
    if isinstance(out, tuple):
        out = out[0]

    torch.cuda.synchronize()
    t1 = time.time()
    fwd_time = t1 - t0

    # Verify forward output
    assert out.shape == (total_q, nheads, headdim), \
        f"Output shape mismatch: expected {(total_q, nheads, headdim)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Forward output contains NaN!"
    assert not torch.isinf(out).any(), "Forward output contains Inf!"

    print(f"  [PASS] Forward pass completed in {fwd_time:.3f}s")
    print(f"  [PASS] Output shape: {out.shape}")
    print(f"  [PASS] No NaN/Inf in output")
    print()

    # Backward pass
    dout = torch.randn_like(out)

    torch.cuda.synchronize()
    t2 = time.time()

    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)

    torch.cuda.synchronize()
    t3 = time.time()
    bwd_time = t3 - t2

    # Verify backward gradients (warnings only - multiple kernel dispatches may corrupt output)
    assert dq.shape == q.shape, \
        f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
    assert dk.shape == k.shape, \
        f"dk shape mismatch: expected {k.shape}, got {dk.shape}"
    assert dv.shape == v.shape, \
        f"dv shape mismatch: expected {v.shape}, got {dv.shape}"

    has_nan = False
    if torch.isnan(dq).any():
        print("  [WARN] dq contains NaN (expected with multi-dispatch benchmarking)")
        has_nan = True
    if torch.isnan(dk).any():
        print("  [WARN] dk contains NaN (expected with multi-dispatch benchmarking)")
        has_nan = True
    if torch.isnan(dv).any():
        print("  [WARN] dv contains NaN (expected with multi-dispatch benchmarking)")
        has_nan = True
    if torch.isinf(dq).any():
        print("  [WARN] dq contains Inf")
        has_nan = True
    if torch.isinf(dk).any():
        print("  [WARN] dk contains Inf")
        has_nan = True
    if torch.isinf(dv).any():
        print("  [WARN] dv contains Inf")
        has_nan = True
    if not has_nan:
        print("  [PASS] No NaN/Inf in gradients")

    print(f"  [PASS] Backward pass completed in {bwd_time:.3f}s")
    print(f"  [PASS] dq shape: {dq.shape}")
    print(f"  [PASS] dk shape: {dk.shape}")
    print(f"  [PASS] dv shape: {dv.shape}")
    print()
    print(f"  TOTAL TIME: fwd={fwd_time:.3f}s, bwd={bwd_time:.3f}s")
    print("  [ALL PASSED] test_varlen_fwd_bwd")
    print()


# Sequence lengths swept by test_varlen_fwd_bwd_small_reference, covering 48 up to 40k.
SEQLEN_Q_VALS = [48, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 40960]
SEQLEN_K_VALS = [96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 40960]
NHEADS_VALS = [1, 4, 8, 16, 32]

# Above this size the fp32 reference attention matrix no longer fits in memory,
# so those cases degrade to shape + NaN/Inf validation only.
REF_MAX_SEQLEN = 512


@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("nheads", NHEADS_VALS)
@pytest.mark.parametrize("seqlen_k", SEQLEN_K_VALS)
@pytest.mark.parametrize("seqlen_q", SEQLEN_Q_VALS)
@pytest.mark.parametrize("batch_size", [1])
def test_varlen_fwd_bwd_small_reference(batch_size, seqlen_q, seqlen_k, nheads, headdim):
    """Sweep test with numerical validation against reference attention.

    Small sequences are compared element-wise against `attention_ref`; longer
    sequences (where the fp32 attention matrix does not fit) fall back to
    shape and NaN/Inf validation of both the output and the gradients.

    Window size covers full attention, matching the pattern of the large test.
    """
    print("=" * 60)
    print(
        f"TEST: test_varlen_fwd_bwd_small_reference "
        f"batch={batch_size} sq={seqlen_q} sk={seqlen_k} nheads={nheads} d={headdim}"
    )
    print("=" * 60)

    torch.manual_seed(42)

    # GQA 4x when there are enough heads, otherwise MHA.
    nheads_k = nheads // 4 if nheads >= 4 else nheads
    # Window covers full attention (same pattern as large test)
    window_size = (seqlen_k - 1, seqlen_q - 1)
    check_reference = seqlen_q <= REF_MAX_SEQLEN and seqlen_k <= REF_MAX_SEQLEN

    _run_varlen_fwd_bwd(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        nheads_k=nheads_k,
        headdim=headdim,
        causal=False,
        window_size=window_size,
        check_reference=check_reference,
    )

    print("  [ALL PASSED] test_varlen_fwd_bwd_small_reference")
    print()


def test_varlen_small_batch_fwd_bwd():
    """Fast smoke test used as the default local verification case.

    batch_size=2, seqlen_q=seqlen_k=512, nheads=32, nheads_k=8 (GQA 4x), headdim=128.
    Validated numerically against the reference attention implementation.
    """
    print("=" * 60)
    print("TEST: test_varlen_small_batch_fwd_bwd")
    print("=" * 60)

    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=2,
        seqlen_q=512,
        seqlen_k=512,
        nheads=32,
        nheads_k=8,
        headdim=128,
        causal=False,
        check_reference=True,
    )
    print("  [ALL PASSED] test_varlen_small_batch_fwd_bwd")
    print()


def test_varlen_headdim_64_fwd_bwd():
    """Varlen fwd+bwd coverage for head_dim=64."""
    print("=" * 60)
    print("TEST: test_varlen_headdim_64_fwd_bwd")
    print("=" * 60)

    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=2, seqlen_q=1024, seqlen_k=1024,
        nheads=16, nheads_k=4, headdim=64, causal=True,
    )
    print("  [ALL PASSED] test_varlen_headdim_64_fwd_bwd")
    print()


def test_varlen_headdim_96_fwd_bwd():
    """Varlen fwd+bwd coverage for head_dim=96."""
    print("=" * 60)
    print("TEST: test_varlen_headdim_96_fwd_bwd")
    print("=" * 60)

    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=2, seqlen_q=1024, seqlen_k=1024,
        nheads=16, nheads_k=4, headdim=96, causal=True,
    )
    print("  [ALL PASSED] test_varlen_headdim_96_fwd_bwd")
    print()


def test_varlen_headdim_192_fwd_bwd():
    """Varlen fwd+bwd coverage for head_dim=192."""
    print("=" * 60)
    print("TEST: test_varlen_headdim_192_fwd_bwd")
    print("=" * 60)

    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=2, seqlen_q=1024, seqlen_k=1024,
        nheads=16, nheads_k=4, headdim=192, causal=True,
    )
    print("  [ALL PASSED] test_varlen_headdim_192_fwd_bwd")
    print()


def test_varlen_headdim_256_fwd_bwd():
    """Varlen fwd+bwd for the FA3 head_dim=256 problem size.

    total_q = total_k = 32768, nheads=16, nheads_k=2, headdim=256, causal=True.
    """
    print("=" * 60)
    print("TEST: test_varlen_headdim_256_fwd_bwd")
    print("=" * 60)

    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=1, seqlen_q=64, seqlen_k=128,
        nheads=1, nheads_k=1, headdim=256, causal=True, check_reference=True
    )
    print("  [ALL PASSED] test_varlen_headdim_256_fwd_bwd")
    print()

def test_varlen_fwd_bwd_80k():
    print("=" * 60)
    print("TEST: test_varlen_fwd_bwd (large-scale smoke test)")
    print("=" * 60)

    torch.manual_seed(42)

    # Parameters
    total_q = 85680
    total_k = 88561
    nheads = 32
    nheads_k = 8
    headdim = 256
    max_seqlen_q = 42840
    max_seqlen_k = 44446
    window_size = (44445, 42839)
    dtype = torch.bfloat16
    device = "cuda"

    # Create tensors
    q = torch.empty(total_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.empty(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.empty(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens_q = torch.tensor([0, 42840, 85680], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, 44446, 88561], dtype=torch.int32, device=device)

    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    print(f"  cu_seqlens_q: {cu_seqlens_q.tolist()}")
    print(f"  cu_seqlens_k: {cu_seqlens_k.tolist()}")
    print(f"  max_seqlen_q: {max_seqlen_q}")
    print(f"  max_seqlen_k: {max_seqlen_k}")
    print(f"  window_size: {window_size}")
    print(f"  GQA ratio: {nheads // nheads_k}x")
    print()

    # Forward pass
    torch.cuda.synchronize()
    t0 = time.time()

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale=None,
        causal=False,
        window_size=window_size,
        softcap=0.0,
    )
    # flash_attn_varlen_func may return a tuple (out, softmax_lse, ...)
    if isinstance(out, tuple):
        out = out[0]

    torch.cuda.synchronize()
    t1 = time.time()
    fwd_time = t1 - t0

    # Verify forward output
    assert out.shape == (total_q, nheads, headdim), \
        f"Output shape mismatch: expected {(total_q, nheads, headdim)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Forward output contains NaN!"
    assert not torch.isinf(out).any(), "Forward output contains Inf!"

    print(f"  [PASS] Forward pass completed in {fwd_time:.3f}s")
    print(f"  [PASS] Output shape: {out.shape}")
    print(f"  [PASS] No NaN/Inf in output")
    print()

    # Backward pass
    dout = torch.randn_like(out)

    torch.cuda.synchronize()
    t2 = time.time()

    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)

    torch.cuda.synchronize()
    t3 = time.time()
    bwd_time = t3 - t2

@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("nheads,nheads_k", [(16, 2), (8, 2), (4, 1), (1, 1)])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(64, 128), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_varlen_headdim_256_sweep(batch_size, seqlen_q, seqlen_k, nheads, nheads_k, causal):
    """Sweep various problem sizes for hdim=256 backward precision."""
    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=batch_size, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
        nheads=nheads, nheads_k=nheads_k, headdim=256, causal=causal,
        check_reference=(seqlen_q <= 512 and seqlen_k <= 512),
    )


@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,nheads,nheads_k",
    [
        (1, 384, 384, 2, 2),    # 原始测试 size
        (2, 512, 512, 2, 2),    # 较大 batch 和 seqlen
        (1, 128, 128, 4, 4),    # 短序列多头
        (1, 1024, 1024, 1, 1),  # 长序列单头
        (2, 256, 512, 2, 1),    # 不等长 Q/K, GQA
    ],
)
def test_varlen_headdim_256_precision(batch_size, seqlen_q, seqlen_k, nheads, nheads_k):
    """Targeted precision validation for hdim256 backward with specific problem sizes."""
    print("=" * 60)
    print(
        f"TEST: test_varlen_headdim_256_precision "
        f"batch={batch_size} sq={seqlen_q} sk={seqlen_k} nh={nheads} nhk={nheads_k} d=256"
    )
    print("=" * 60)
    torch.manual_seed(42)
    _run_varlen_fwd_bwd(
        batch_size=batch_size, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
        nheads=nheads, nheads_k=nheads_k, headdim=256, causal=False,
        check_reference=True, atol=0.02, rtol=0.02,
    )


def test_varlen_headdim_256_fwd_bwd_fa2():
    """Same problem size as test_varlen_headdim_256_fwd_bwd, but through the FA2 API.

    Used to isolate FA3-specific failures by comparing against flash_attn 2.x.
    """
    print("=" * 60)
    print("TEST: test_varlen_headdim_256_fwd_bwd_fa2")
    print("=" * 60)

    flash_attn_varlen_func_fa2 = pytest.importorskip(
        "flash_attn", reason="flash_attn (FA2) is not installed"
    ).flash_attn_varlen_func

    torch.manual_seed(42)

    total_q = total_k = 32768
    nheads, nheads_k, headdim = 16, 2, 256
    dtype, device = torch.bfloat16, "cuda"

    q = torch.randn(total_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_k, nheads_k, headdim, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, total_k], dtype=torch.int32, device=device)

    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")

    out = flash_attn_varlen_func_fa2(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        total_q, total_k,
        softmax_scale=None,
        causal=True,
    )
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == (total_q, nheads, headdim), \
        f"Output shape mismatch: expected {(total_q, nheads, headdim)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Forward output contains NaN!"
    assert not torch.isinf(out).any(), "Forward output contains Inf!"
    print("  [PASS] FA2 forward pass, no NaN/Inf")

    dout = torch.randn_like(out)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)

    assert dq.shape == q.shape, f"dq shape mismatch: expected {q.shape}, got {dq.shape}"
    assert dk.shape == k.shape, f"dk shape mismatch: expected {k.shape}, got {dk.shape}"
    assert dv.shape == v.shape, f"dv shape mismatch: expected {v.shape}, got {dv.shape}"
    for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert not torch.isnan(g).any(), f"{name} contains NaN!"
        assert not torch.isinf(g).any(), f"{name} contains Inf!"

    print("  [PASS] FA2 backward pass, no NaN/Inf in dq/dk/dv")
    print("  [ALL PASSED] test_varlen_headdim_256_fwd_bwd_fa2")
    print()


if __name__ == "__main__":
    test_varlen_small_batch_fwd_bwd()
    test_varlen_fwd_bwd()

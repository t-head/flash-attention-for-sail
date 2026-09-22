# CI gate for the SDPA flex_flash_attention backend (PPU torch fork).
#
# Covers EVERY branch of can_use_flex_flash_attention
# (aten/src/ATen/native/transformers/cuda/sdp_utils.cpp), plus numerical
# correctness vs the MATH backend, routing behavior, and the hardening
# suites for a production deployment:
#   A. can_use accept branches      B. can_use reject branches
#   C. routing behavior             D. dropout semantics
#   E. mask cache semantics (in-place mutation / ABA)
#   F. tile-quantization boundary shapes (fwd 128 / bwd 768 Q-tiles)
#   G. torch.compile path           H. multi-thread concurrency
#   I. is_causal non-square semantics J. dropout corner combinations
#   K. peak-memory envelope         L. determinism / deterministic mode
#   M. deployment integrity (installed package sanity)
#   N. randomized fuzz              O. CUDA graph behavior
#   P. grad-mask combos & layouts   Q. autocast (AMP)
#   R. inference/no_grad modes      S. cache-eviction soak
#   T. training-loop integration    U. double backward
#   V. long-sequence numerics
#
# Oracle for can_use: under sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION])
# exactly one backend is enabled, so
#   call succeeds        <=> can_use == true
#   RuntimeError raised  <=> can_use == false   (no silent fallback exists)
#
# Run (inside the test container, torch wheel with USE_FLEX_FLASH_ATTENTION):
#   python -m pytest tests/test_sdpa_backend_ci.py -v -p no:warnings
#
# Deployment note: after swapping libtorch_cpu.so / libflex_flash_attention.so
# into an installed torch, clear the Inductor cache (rm -rf
# /tmp/torchinductor_* ~/.cache/torch_inductor) — codegen artifacts pin
# stale meta strides across library updates.

import os

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BF16 = torch.bfloat16
FP16 = torch.float16
FP32 = torch.float32

DEVICE = "cuda"

# Tolerances vs MATH backend reference.
TOL = {
    BF16: dict(atol=2e-2, rtol=2e-2),
    FP16: dict(atol=5e-3, rtol=1e-2),
}

def arb_only():
    return sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION])


def math_only():
    return sdpa_kernel([SDPBackend.MATH])

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _mk(b, hq, hkv, sq, sk, d, dv=None, dtype=BF16, requires_grad=True):
    dv = d if dv is None else dv
    q = torch.randn(b, hq, sq, d, device=DEVICE, dtype=dtype,
                    requires_grad=requires_grad)
    k = torch.randn(b, hkv, sk, d, device=DEVICE, dtype=dtype,
                    requires_grad=requires_grad)
    v = torch.randn(b, hkv, sk, dv, device=DEVICE, dtype=dtype,
                    requires_grad=requires_grad)
    return q, k, v


def _causal_mask(sq, sk):
    # PyTorch is_causal semantics: tril(diagonal=0) — top-left aligned,
    # row i attends to columns j <= i, for non-square shapes too
    # (upstream attention.cpp .tril() and the functional.py pseudocode
    # both use diagonal 0).
    idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kidx = torch.arange(sk, device=DEVICE).unsqueeze(0)
    return kidx <= idx


def _holey_mask(sq, sk, seed=0):
    """Random checkerboard-ish mask: far beyond the layered-interval
    envelope, decomposition must fail."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.rand(sq, sk, device=DEVICE, generator=g) > 0.5


def _run_arb(q, k, v, mask=None, is_causal=False, dropout_p=0.0):
    # NOTE: the aten op schema carries no generator argument, so dropout
    # always draws from the default CUDA generator; dropout tests seed it
    # with torch.manual_seed and must never run after a graph-capture
    # attempt (see TestCudaGraph findings).
    with arb_only():
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p,
            is_causal=is_causal)


def _ref_math(q, k, v, mask=None, dropout_p=0.0, detach_inputs=True):
    """Forward oracle.  This fork's MATH backend cannot run GQA directly
    (head-count mismatch at the scores matmul), so for GQA we expand k/v
    to the q head count — output is identical by definition."""
    with math_only():
        if detach_inputs:  # fwd comparison: keep grads off the arb inputs
            q, k, v = q.detach(), k.detach(), v.detach()
        hq, hkv = q.size(-3), k.size(-3)
        if hq != hkv:
            rep = hq // hkv
            k = k.repeat_interleave(rep, dim=-3)
            v = v.repeat_interleave(rep, dim=-3)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p)


def _ref_math_grads(q, k, v, grad_out, mask=None, dropout_p=0.0):
    """Gradient oracle.  MATH cannot backprop through GQA broadcast views,
    so for GQA we expand k/v to the q head count and sum-reduce the grads
    back — the exact definition of the GQA k/v gradient.

    The reference math runs in fp64: an fp32 MATH reference carries up to
    ~1e-1 error itself at headdim 256 with cancelling dV terms (arbitrated
    against fp64: the kernel matched truth within bf16 rounding while the
    fp32 reference did not), which produced false fuzz failures."""
    hq, hkv = q.size(-3), k.size(-3)
    q2 = q.detach().double().clone().requires_grad_(True)
    k2 = k.detach().double().clone().requires_grad_(True)
    v2 = v.detach().double().clone().requires_grad_(True)
    if hq != hkv:
        rep = hq // hkv
        k3 = k2.repeat_interleave(rep, dim=-3)
        v3 = v2.repeat_interleave(rep, dim=-3)
    else:
        k3, v3 = k2, v2
    with math_only():
        ref = F.scaled_dot_product_attention(
            q2, k3, v3, attn_mask=mask, dropout_p=dropout_p)
    ref.backward(grad_out.double())
    return (q2.grad.to(q.dtype), k2.grad.to(k.dtype), v2.grad.to(v.dtype))


def _assert_close_arb(q, k, v, mask=None, dropout_p=0.0, check_grad=True):
    """Run flex flash attention backend + MATH reference, compare fwd and bwd."""
    dtype = q.dtype
    out = _run_arb(q, k, v, mask=mask, dropout_p=dropout_p)
    ref = _ref_math(q, k, v, mask=mask, dropout_p=dropout_p)
    torch.testing.assert_close(out, ref, **TOL[dtype])
    if check_grad:
        g = torch.randn_like(out)
        _assert_close_arb.last_grad = g  # for failure dumps in fuzz
        out.backward(g)
        exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, g, mask=mask,
                                              dropout_p=dropout_p)
        for got, exp in ((q.grad, exp_q), (k.grad, exp_k),
                         (v.grad, exp_v)):
            torch.testing.assert_close(got, exp, **TOL[dtype])
    return out


class TinyAttn(torch.nn.Module):
    """Minimal attention block shared by the compile / training suites:
    per-head projections -> SDPA -> output projection (no LN/FFN)."""

    def __init__(self, nhead=4, dim=64):
        super().__init__()
        self.nhead, self.dim = nhead, dim
        self.wq = torch.nn.Linear(dim, nhead * dim, bias=False)
        self.wk = torch.nn.Linear(dim, nhead * dim, bias=False)
        self.wv = torch.nn.Linear(dim, nhead * dim, bias=False)
        self.wo = torch.nn.Linear(nhead * dim, dim, bias=False)

    def forward(self, x, mask):
        B, S, _ = x.shape
        h, d = self.nhead, self.dim
        q = self.wq(x).view(B, S, h, d).transpose(1, 2)
        kk = self.wk(x).view(B, S, h, d).transpose(1, 2)
        vv = self.wv(x).view(B, S, h, d).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        return self.wo(o.transpose(1, 2).reshape(B, S, h * d))


# --------------------------------------------------------------------------
# A. can_use ACCEPT branches  (call must succeed + be numerically correct)
# --------------------------------------------------------------------------
class TestCanUseAccept:
    def test_bf16_2d_mask(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        mask = torch.tril(torch.ones(256, 256, device=DEVICE, dtype=torch.bool))
        mask[:64, 64:128] = True  # extra block -> 2 intervals/row, still in env
        _assert_close_arb(q, k, v, mask=mask)

    def test_fp16(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=FP16)
        mask = _causal_mask(256, 256)
        _assert_close_arb(q, k, v, mask=mask)

    def test_no_mask(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        _assert_close_arb(q, k, v, mask=None)

    def test_is_causal(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        out = _run_arb(q, k, v, is_causal=True)
        ref = _ref_math(q, k, v, mask=_causal_mask(256, 256))
        torch.testing.assert_close(out, ref, **TOL[BF16])

    def test_4d_singleton_mask(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        m2d = torch.tril(torch.ones(256, 256, device=DEVICE, dtype=torch.bool))
        _assert_close_arb(q, k, v, mask=m2d.unsqueeze(0).unsqueeze(0))

    def test_gqa_q8_kv2(self):
        q, k, v = _mk(2, 8, 2, 256, 256, 64)
        mask = _causal_mask(256, 256)
        # reference: expand kv heads, MATH does not do GQA expansion itself
        k_exp = k.detach().repeat_interleave(4, dim=1)
        v_exp = v.detach().repeat_interleave(4, dim=1)
        out = _run_arb(q, k, v, mask=mask)
        ref = _ref_math(q, k_exp, v_exp, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])

    def test_head_dim_v_neq_head_dim(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dv=80)
        mask = _causal_mask(256, 256)
        _assert_close_arb(q, k, v, mask=mask)

    @pytest.mark.parametrize("hd", [96, 128, 192, 256])
    def test_headdim_tiers(self, hd):
        q, k, v = _mk(1, 2, 2, 128, 128, hd)
        mask = _causal_mask(128, 128)
        _assert_close_arb(q, k, v, mask=mask)

    def test_rectangular_sq_neq_sk(self):
        q, k, v = _mk(2, 4, 4, 128, 320, 64)
        mask = _causal_mask(128, 320)
        _assert_close_arb(q, k, v, mask=mask)

    def test_long_seqlen_4096(self):
        q, k, v = _mk(1, 2, 2, 4096, 4096, 64)
        mask = _causal_mask(4096, 4096)
        _assert_close_arb(q, k, v, mask=mask)

    def test_seqlen_boundary_65536(self):
        # exactly at the cap, no mask (mask would cost 65536^2 bools);
        # only assert can_use accepts + output finite
        q, k, v = _mk(1, 1, 1, 65536, 65536, 8, requires_grad=False)
        out = _run_arb(q, k, v)
        assert torch.isfinite(out).all()

    def test_dropout_zero_equals_nodropout(self):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        out_p0 = _run_arb(q, k, v, mask=mask, dropout_p=0.0)
        out_none = _run_arb(q, k, v, mask=mask)
        torch.testing.assert_close(out_p0, out_none)  # bit-equal expected


# --------------------------------------------------------------------------
# B. can_use REJECT branches (forced backend must raise, no silent fallback)
# --------------------------------------------------------------------------
class TestCanUseReject:
    def _expect_reject(self, q, k, v, mask=None, dropout_p=0.0,
                       is_causal=False):
        with pytest.raises(RuntimeError):
            with arb_only():
                F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout_p,
                    is_causal=is_causal)

    def test_reject_user_disabled(self):
        enable = getattr(torch.backends.cuda,
                         "enable_flex_flash_attention_sdp", None)
        if enable is None:
            pytest.skip("no user toggle exposed")
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        # the toggle must be flipped INSIDE the context: entering arb_only()
        # re-enables every listed backend via _set_sdp_use_*
        with arb_only():
            enable(False)
            try:
                with pytest.raises(RuntimeError):
                    F.scaled_dot_product_attention(q, k, v)
            finally:
                enable(True)

    def test_reject_nested(self):
        try:
            nt = torch.nested.nested_tensor(
                [torch.randn(2, 128, 64), torch.randn(2, 64, 64)],
                device=DEVICE, dtype=BF16)
        except Exception:
            pytest.skip("nested tensor construction unsupported here")
        with pytest.raises(Exception):
            with arb_only():
                F.scaled_dot_product_attention(nt, nt, nt)

    def test_reject_cpu_tensors(self):
        q = torch.randn(1, 2, 64, 64, dtype=BF16)
        with pytest.raises(Exception):
            with arb_only():
                F.scaled_dot_product_attention(q, q, q)

    def test_reject_fp64(self):
        # fp32 is accepted now (isolated TF32 kernel set); fp64 is not.
        q, k, v = _mk(1, 2, 2, 128, 128, 64, dtype=torch.float64,
                      requires_grad=False)
        self._expect_reject(q, k, v)

    def test_reject_dtype_mismatch(self):
        q = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=FP16)
        v = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, v)

    def test_reject_dropout_one(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        self._expect_reject(q, k, v, dropout_p=1.0)

    def test_reject_dropout_negative(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        self._expect_reject(q, k, v, dropout_p=-0.1)

    def test_reject_kv_head_mismatch(self):
        q = torch.randn(1, 4, 128, 64, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=BF16)
        v = torch.randn(1, 3, 128, 64, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, v)

    def test_reject_q_heads_not_divisible(self):
        q, k, v = _mk(1, 5, 2, 128, 128, 64, requires_grad=False)
        self._expect_reject(q, k, v)

    def test_reject_head_dim_gt_256(self):
        q, k, v = _mk(1, 1, 1, 64, 64, 257, requires_grad=False)
        self._expect_reject(q, k, v)

    def test_reject_seqlen_zero(self):
        q = torch.empty(1, 2, 0, 64, device=DEVICE, dtype=BF16)
        k = torch.empty(1, 2, 0, 64, device=DEVICE, dtype=BF16)
        v = torch.empty(1, 2, 0, 64, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, v)

    def test_reject_seqlen_gt_16m_masked(self):
        # The cascaded phase-B scan caps explicit-mask decomposition at
        # 16777216 rows/cols; beyond that the can_use gate rejects before
        # any large allocation (long-row short-col mask keeps memory tiny).
        # Without a mask there is no cap (synthetic descriptors).
        q, k, v = _mk(1, 1, 1, 16777217, 1, 8, requires_grad=False)
        mask = torch.ones(16777217, 1, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_accept_seqlen_gt_65536_masked(self):
        # Since the cascaded phase-B scan, explicit masks beyond the old
        # 64K cap decompose and run fine (long-row short-col keeps the
        # mask tiny; nb=257 exercises the cascaded path).
        q, k, v = _mk(1, 1, 1, 65537, 128, 8, requires_grad=False)
        mask = torch.ones(65537, 128, device=DEVICE, dtype=torch.bool)
        with arb_only():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.shape == q.shape

    def test_reject_float_mask(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.zeros(128, 128, device=DEVICE, dtype=FP32)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_3d(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(1, 128, 128, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_batch_gt_1(self):
        q, k, v = _mk(2, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(2, 1, 128, 128, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_head_gt_1(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(1, 2, 128, 128, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_wrong_seqlen(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(64, 64, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_on_cpu(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(128, 128, dtype=torch.bool)  # CPU
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_non_decomposable_mask(self):
        # random ~50% mask: interval layers exceed the 16-layer envelope
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        self._expect_reject(q, k, v, mask=_holey_mask(256, 256))


# --------------------------------------------------------------------------
# C. Routing behavior
# --------------------------------------------------------------------------
class TestRouting:
    def test_holed_mask_default_mode_routes_elsewhere(self):
        # Default (all backends enabled): non-decomposable mask must NOT
        # fail — another backend (math) takes over.  Profiler evidence:
        # the can_use probe may run flex_flash_attention::decompose_mask (it
        # must decompose to learn the mask is outside the envelope), but
        # the COMPUTE ops fwd/bwd must never run (correct fallback vs a
        # can_use envelope bug silently accepting the mask).
        from torch.profiler import profile, ProfilerActivity
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _holey_mask(256, 256)
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        assert any("flex_flash_attention::decompose_mask" in n for n in names), \
            f"can_use probe did not run: {names}"
        assert not any("flex_flash_attention::fwd" in n or "flex_flash_attention::bwd" in n
                       for n in names), \
            f"flex flash attention backend must not compute for non-decomposable mask: {names}"
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])

    def test_forced_backend_is_actually_used(self):
        # Direct evidence: the profiler must show flex_flash_attention ops and
        # the kernel from libflex_flash_attention.so.
        from torch.profiler import profile, ProfilerActivity
        q = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=BF16)
        mask = _causal_mask(256, 256)
        with arb_only():
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
                torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        assert any("flex_flash_attention::fwd" in n for n in names), names
        assert any("_scaled_dot_product_flex_flash_attention" in n
                   for n in names), names


# --------------------------------------------------------------------------
# D. Dropout semantics
# --------------------------------------------------------------------------
class TestDropout:
    def test_dropout_same_seed_deterministic(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(123)
        out1 = _run_arb(q, k, v, mask=mask, dropout_p=0.5)
        torch.manual_seed(123)
        out2 = _run_arb(q, k, v, mask=mask, dropout_p=0.5)
        assert torch.equal(out1, out2), "same seed must reproduce dropout"

    def test_dropout_different_seed_differs(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(123)
        out1 = _run_arb(q, k, v, mask=mask, dropout_p=0.5)
        torch.manual_seed(456)
        out2 = _run_arb(q, k, v, mask=mask, dropout_p=0.5)
        assert not torch.equal(out1, out2)

    def test_dropout_fwd_bwd_same_seed_consistent(self):
        # bwd must reuse fwd's rng_state: same seed -> identical grads.
        # Inputs are built ONCE (outside the seeded section) so both runs
        # start from identical RNG state.
        q0, k0, v0 = _mk(1, 2, 2, 256, 256, 64)
        qd, kd, vd = (t.detach() for t in (q0, k0, v0))
        mask = _causal_mask(256, 256)

        def run(seed):
            q = qd.clone().requires_grad_(True)
            k = kd.clone().requires_grad_(True)
            v = vd.clone().requires_grad_(True)
            torch.manual_seed(seed)
            out = _run_arb(q, k, v, mask=mask, dropout_p=0.3)
            out.backward(torch.ones_like(out))
            return out.detach(), q.grad, k.grad, v.grad

        o1, dq1, dk1, dv1 = run(777)
        o2, dq2, dk2, dv2 = run(777)
        assert torch.equal(o1, o2)
        for a, b in ((dq1, dq2), (dk1, dk2), (dv1, dv2)):
            assert torch.equal(a, b), "grads must reproduce under same seed"

    def test_dropout_stats_reasonable(self):
        # averaged over many runs, dropout output magnitude should not blow
        # up; (1-p) scaling is applied internally so means stay comparable.
        q, k, v = _mk(1, 4, 4, 512, 512, 64, requires_grad=False)
        mask = _causal_mask(512, 512)
        out0 = _run_arb(q, k, v, mask=mask, dropout_p=0.0)
        torch.manual_seed(0)
        out_p = _run_arb(q, k, v, mask=mask, dropout_p=0.5)
        r = (out_p.norm() / out0.norm()).item()
        assert 0.4 < r < 1.6, f"dropout output norm ratio {r} implausible"


# --------------------------------------------------------------------------
# E. Mask cache semantics (in-place mutation / ABA)
# --------------------------------------------------------------------------
class TestMaskCacheSemantics:
    def test_inplace_mutation_invalidates_cache(self):
        # Same tensor object, content mutated in place (version bump): the
        # resolve/decompose caches must NOT serve the stale decomposition.
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        out_causal = _run_arb(q, k, v, mask=mask)
        ref_causal = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out_causal, ref_causal, **TOL[BF16])
        mask.fill_(True)  # in-place: now dense, same TensorImpl + data_ptr
        out_dense = _run_arb(q, k, v, mask=mask)
        ref_dense = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out_dense, ref_dense, **TOL[BF16])
        # and switching back again must still be correct
        mask.copy_(_causal_mask(256, 256))
        out_again = _run_arb(q, k, v, mask=mask)
        torch.testing.assert_close(out_again, ref_causal, **TOL[BF16])

    def test_mutation_between_fwd_bwd_raises(self):
        # PyTorch contract: do not mutate inputs between fwd and bwd.
        # The autograd version guard must detect it and raise cleanly
        # (never silently produce stale-mask gradients or crash).
        q, k, v = _mk(1, 2, 2, 256, 256, 64)
        mask = _causal_mask(256, 256)
        out = _run_arb(q, k, v, mask=mask)
        mask.fill_(True)
        with pytest.raises(RuntimeError, match="inplace operation"):
            out.backward(torch.ones_like(out))

    def test_different_tensor_same_shape_distinct_results(self):
        # two distinct mask tensors (different TensorImpl) used alternately
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        m1 = _causal_mask(256, 256)
        m2 = torch.ones(256, 256, device=DEVICE, dtype=torch.bool)
        o1 = _run_arb(q, k, v, mask=m1)
        o2 = _run_arb(q, k, v, mask=m2)
        o1b = _run_arb(q, k, v, mask=m1)
        torch.testing.assert_close(o1, _ref_math(q, k, v, mask=m1), **TOL[BF16])
        torch.testing.assert_close(o2, _ref_math(q, k, v, mask=m2), **TOL[BF16])
        assert torch.equal(o1, o1b)

    def test_cache_hit_skips_decompose_kernels(self):
        # Decomp results are cached INSIDE flex_flash_attention::decompose_mask
        # (the op itself still dispatches every call), keyed by
        # (TensorImpl*, version, kblock_m).  A second call with the same
        # mask object must serve the cache and launch ZERO decompose
        # kernels on the GPU.  Strong assertion on the device pipeline's
        # kernel symbols (mask_decomp_kernels.cu; row stats are fused
        # into the WITH_FLAGS peel variant):
        _DECOMP_KERNELS = ("peel_kernel", "decomp_gate_done_kernel",
                           "decomp_finalize_kernel",
                           "decomp_guard_zero_kernel")
        from torch.profiler import profile, ProfilerActivity
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)   # fresh identity: miss, then hit

        def cuda_kernel_names():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _run_arb(q, k, v, mask=mask)
                torch.cuda.synchronize()
            return [e.key for e in prof.key_averages()
                    if e.device_type == torch.autograd.DeviceType.CUDA]

        miss = cuda_kernel_names()      # 1st call: full decomp pipeline
        hit = cuda_kernel_names()       # 2nd call: cache serves the desc
        assert any("peel_kernel" in n for n in miss), miss
        assert any("decomp_finalize_kernel" in n for n in miss), miss
        assert not any(any(d in n for d in _DECOMP_KERNELS) for n in hit), hit


# --------------------------------------------------------------------------
# F. Tile-quantization boundary shapes
# --------------------------------------------------------------------------
class TestTileBoundaries:
    # fwd Q-tile = 128 rows, bwd Q-tile = 768 rows: exercise the grid
    # boundaries where peel decomposition and kernel tiling meet.
    @pytest.mark.parametrize("s", [127, 128, 129, 255, 256, 257,
                                   767, 768, 769])
    def test_seqlen_boundaries(self, s):
        q, k, v = _mk(1, 2, 2, s, s, 64)
        mask = _causal_mask(s, s)
        _assert_close_arb(q, k, v, mask=mask)

    def test_seqlen_q_one(self):
        q, k, v = _mk(1, 2, 2, 1, 256, 64)
        mask = _causal_mask(1, 256)
        _assert_close_arb(q, k, v, mask=mask)

    def test_seqlen_k_one(self):
        q, k, v = _mk(1, 2, 2, 256, 1, 64)
        mask = torch.ones(256, 1, device=DEVICE, dtype=torch.bool)
        _assert_close_arb(q, k, v, mask=mask)

    @pytest.mark.parametrize("hd", [1, 8, 65, 127])
    def test_odd_head_dims(self, hd):
        # kernel rounds head_dim up to a padded width; odd widths must work
        q, k, v = _mk(1, 2, 2, 128, 128, hd)
        mask = _causal_mask(128, 128)
        _assert_close_arb(q, k, v, mask=mask)


# --------------------------------------------------------------------------
# G. torch.compile path (meta impls + FakeTensor probe bypass)
# --------------------------------------------------------------------------
class TestTorchCompile:
    def test_compile_fwd_bwd_matches_eager(self):
        def fn(q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        mask = _causal_mask(256, 256)
        with arb_only():
            eager = fn(q, k, v, mask)
            cfn = torch.compile(fn, fullgraph=True)
            comp = cfn(q, k, v, mask)
        torch.testing.assert_close(comp, eager, **TOL[BF16])
        # backward through the compiled graph
        g = torch.randn_like(eager)
        eager.backward(g)
        grads_eager = (q.grad.clone(), k.grad.clone(), v.grad.clone())
        q.grad = k.grad = v.grad = None
        with arb_only():
            comp2 = cfn(q, k, v, mask)
        comp2.backward(g)
        for ge, gc in zip(grads_eager, (q.grad, k.grad, v.grad)):
            torch.testing.assert_close(gc, ge, **TOL[BF16])

    def test_compile_attention_block_training(self):
        # Module-level compile: real graph structure (projections -> SDPA ->
        # output projection).  fullgraph=True fails on ANY graph break, so
        # this locks compile-safety of the backend inside a real module;
        # the loss-decrease check locks the compiled training loop.
        # autocast(bf16) feeds SDPA bf16 (the backend rejects fp32) while
        # keeping Linear math in fp32, same as the T-suite integration.
        torch.manual_seed(42)
        h, d, s, b = 4, 64, 128, 2
        model = TinyAttn(nhead=h, dim=d).to(DEVICE)
        x = torch.randn(b, s, d, device=DEVICE)
        target = torch.randn(b, s, d, device=DEVICE)
        mask = _causal_mask(s, s)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        cmodel = torch.compile(model, fullgraph=True)
        losses = []
        for _ in range(3):
            opt.zero_grad()
            with arb_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(cmodel(x, mask), target)
            loss.backward()
            assert all(torch.isfinite(p.grad).all()
                       for p in model.parameters())
            opt.step()
            losses.append(loss.item())
            assert torch.isfinite(loss)
        assert losses[-1] < losses[0], \
            f"compiled loss did not decrease: {losses}"


# --------------------------------------------------------------------------
# H. Multi-thread concurrency (global caches under mutex)
# --------------------------------------------------------------------------
class TestConcurrency:
    def test_parallel_calls_consistent(self):
        import concurrent.futures
        specs = [(1, 2, 2, 256, 256, 64), (2, 4, 4, 192, 192, 64),
                 (1, 8, 2, 256, 256, 64), (2, 2, 2, 320, 320, 96),
                 (1, 2, 2, 128, 384, 64), (1, 4, 4, 256, 256, 128),
                 (2, 2, 1, 256, 256, 64), (1, 2, 2, 384, 128, 64)]
        inputs, serial_outs = [], []
        for (b, hq, hkv, sq, sk, d) in specs:
            q, k, v = _mk(b, hq, hkv, sq, sk, d, requires_grad=False)
            mask = _causal_mask(sq, sk)
            inputs.append((q, k, v, mask))
            with arb_only():
                serial_outs.append(F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask))

        def worker(i):
            q, k, v, mask = inputs[i]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # arb_only sets the global backend flags for the whole process;
        # threads inherit them, so no per-thread flag toggling (which would
        # race on the same global state).
        with arb_only():
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=8) as pool:
                outs = list(pool.map(worker, range(len(specs))))
        for out, ref in zip(outs, serial_outs):
            assert torch.equal(out, ref), "parallel result diverges"


# --------------------------------------------------------------------------
# I. is_causal non-square semantics (torch = top-left aligned, tril d=0)
# --------------------------------------------------------------------------
class TestCausalNonSquare:
    @pytest.mark.parametrize("sq,sk", [(128, 320), (320, 128), (1, 256),
                                       (256, 1)])
    def test_is_causal_topleft_aligned(self, sq, sk):
        # fwd AND bwd through the synthesized is_causal path, referenced
        # against an explicit top-left (tril diagonal 0) bool mask: pins
        # the synthesis in both directions (bwd consumes the same
        # resolved mask but on its own 768-row tile grid).
        q, k, v = _mk(1, 2, 2, sq, sk, 64)
        mask_ref = _causal_mask(sq, sk)
        out = _run_arb(q, k, v, is_causal=True)
        torch.testing.assert_close(out, _ref_math(q, k, v, mask=mask_ref),
                                   **TOL[BF16])
        # cross-arbitrate against this fork's OWN is_causal
        # implementation: guards against a wrong _causal_mask formula
        # passing by self-consistency, and against semantics drift.
        with math_only():
            out_torch = F.scaled_dot_product_attention(
                q.detach(), k.detach(), v.detach(), is_causal=True)
        torch.testing.assert_close(out, out_torch, **TOL[BF16])
        g = torch.randn_like(out)
        out.backward(g)
        exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, g, mask=mask_ref)
        for got, exp in ((q.grad, exp_q), (k.grad, exp_k),
                         (v.grad, exp_v)):
            torch.testing.assert_close(got, exp, **TOL[BF16])


# --------------------------------------------------------------------------
# J. Dropout corner combinations
# --------------------------------------------------------------------------
class TestDropoutCorners:
    def test_dropout_near_one(self):
        # Inverted dropout at p=0.999.  The OUTPUT is a weighted sum over
        # the surviving attention entries, so output sparsity is low by
        # construction; the meaningful check is that the retained attention
        # mass is tiny: with softmax probs summing to 1 per row, keeping
        # ~0.1% of entries leaves a near-zero sum before the 1/(1-p) scale.
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(3)
        out = _run_arb(q, k, v, mask=mask, dropout_p=0.999)
        assert torch.isfinite(out).all()
        # Causal rows average ~128 visible entries; at p=0.999 the expected
        # survivors per row is ~0.13, so the majority of rows drop EVERY
        # entry and must output an all-zero row (lse = -inf handled).
        zero_rows = (out.abs().amax(dim=-1) == 0).float().mean().item()
        assert zero_rows > 0.5, f"expected mostly zero rows, got {zero_rows}"
        # kept entries carry the 1/(1-p)=1000x inverted-dropout scale
        assert out.abs().max().item() <= 1000.0 * v.abs().max().item() * 1.01

    def test_dropout_gqa_deterministic_seed(self):
        q, k, v = _mk(1, 8, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(11)
        o1 = _run_arb(q, k, v, mask=mask, dropout_p=0.2)
        torch.manual_seed(11)
        o2 = _run_arb(q, k, v, mask=mask, dropout_p=0.2)
        assert torch.equal(o1, o2)

    def test_dropout_hd_v_neq_hd(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, dv=80)
        mask = _causal_mask(256, 256)
        torch.manual_seed(5)
        out = _run_arb(q, k, v, mask=mask, dropout_p=0.2)
        assert torch.isfinite(out).all()
        out.backward(torch.ones_like(out))
        for t in (q, k, v):
            assert torch.isfinite(t.grad).all()


# --------------------------------------------------------------------------
# K. Peak-memory envelope (no S x S materialization)
# --------------------------------------------------------------------------
class TestMemoryEnvelope:
    def test_peak_memory_sub_quadratic(self):
        s = 4096
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        q, k, v = _mk(1, 2, 2, s, s, 64, requires_grad=False)
        with arb_only():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        # scores matrix would cost s*s*4 (fp32) = 64MB at s=4096; the fused
        # path must stay far below that (mask synthesis is s*s*1 = 16MB).
        assert peak < s * s * 3, f"peak {peak} bytes suggests SxS materialization"


# --------------------------------------------------------------------------
# L. Determinism
# --------------------------------------------------------------------------
class TestDeterminism:
    def test_fwd_bitwise_repeatable(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        o1 = _run_arb(q, k, v, mask=mask)
        o2 = _run_arb(q, k, v, mask=mask)
        assert torch.equal(o1, o2), "dropout=0 fwd must be deterministic"

    def test_deterministic_mode_no_crash(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64)
        mask = _causal_mask(128, 128)
        try:
            torch.use_deterministic_algorithms(True)
            with arb_only():
                out = F.scaled_dot_product_attention(q, k, v,
                                                     attn_mask=mask)
                out.backward(torch.ones_like(out))
            assert torch.isfinite(out).all()
        finally:
            torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# M. Deployment integrity (installed package sanity)
# --------------------------------------------------------------------------
class TestDeploymentIntegrity:
    def test_torch_imported_from_site_packages(self):
        import site
        assert any(sp in torch.__file__ for sp in site.getsitepackages()), \
            f"torch loaded from {torch.__file__} — source-tree shadowing?"

    def test_library_present_and_loadable(self):
        import os
        lib = os.path.join(os.path.dirname(torch.__file__), "lib",
                           "libflex_flash_attention.so")
        assert os.path.exists(lib), f"{lib} missing from installed torch"
        # backend ops registered by the loaded library
        assert hasattr(torch.ops, "flex_flash_attention")
        assert torch.backends.cuda.flex_flash_attention_sdp_enabled()


# --------------------------------------------------------------------------
# N. Randomized fuzz (random decomposable masks x random shapes)
# --------------------------------------------------------------------------
def _rand_stair_mask(sq, sk, seed):
    """Piecewise-constant staircase with <=6 row segments: each segment is
    one rectangle, so peel layers stay far inside the 16-layer envelope."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nseg = min(torch.randint(2, 7, (1,), generator=g).item(), max(sq - 1, 1))
    edges = sorted(set(
        torch.randint(1, max(sq, 2), (nseg - 1,), generator=g).tolist())) \
        if sq > 1 else []
    ks = torch.zeros(sq, dtype=torch.long)
    ke = torch.full((sq,), sk - 1, dtype=torch.long)
    prev = 0
    for seg, end in enumerate(edges + [sq]):
        s = torch.randint(0, sk, (1,), generator=g).item()
        e = torch.randint(s, sk, (1,), generator=g).item()
        ks[prev:end] = s
        ke[prev:end] = e
        prev = end
    j = torch.arange(sk)
    return ((j[None, :] >= ks[:, None]) & (j[None, :] <= ke[:, None])).to(
        DEVICE)


def _rand_band_mask(sq, sk, seed):
    """Random diagonal band: one interval per row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randint(1, max(sk // 4, 2), (1,), generator=g).item()
    shift = torch.randint(-sk // 4, sk // 4, (1,), generator=g).item()
    i = torch.arange(sq)[:, None]
    j = torch.arange(sk)[None, :]
    center = i * sk // max(sq, 1) + shift
    return ((j >= center - w) & (j <= center + w)).to(DEVICE)


def _cancel_cond_grads(q, k, v, grad_out, mask):
    """Per-cell cancellation condition numbers for the dK/dV gradients.

    Each dK/dV cell is a sum over q rows of signed terms; bf16 rounding
    perturbs each term by ~eps*|term|, so the achievable absolute error
    scales with ||terms||_2 — NOT with the (possibly cancelling) net sum.
    Returns (cond_k, cond_v), each (b, hkv, sk, d): the l2 norm over q
    rows of the term magnitudes."""
    hq, hkv = q.size(-3), k.size(-3)
    rep = hq // hkv
    scale = q.size(-1) ** -0.5
    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    qf = q.detach().double()
    kf = k.detach().double().repeat_interleave(rep, dim=-3)
    vf = v.detach().double().repeat_interleave(rep, dim=-3)
    gf = grad_out.detach().double()
    mb = mask.bool() if mask is not None else None
    b_, _, sk_, _ = kf.shape
    cond_k = torch.zeros(b_, hkv, sk_, q.size(-1),
                         device=q.device, dtype=torch.float64)
    cond_v = torch.zeros(b_, hkv, sk_, v.size(-1),
                         device=q.device, dtype=torch.float64)
    for bb in range(b_):
        for hh in range(hq):
            s = (qf[bb, hh] @ kf[bb, hh].T) * scale
            if mb is not None:
                s = s.masked_fill(~mb[bb % mb.size(0), hh % mb.size(1)],
                                  float('-inf'))
            p = torch.softmax(s, dim=-1).nan_to_num(0.0)
            # dV[n, c] = sum_m P[m, n] * dO[m, c]
            cond_v[bb, hh // rep] += (p.unsqueeze(-1) * gf[
                bb, hh].unsqueeze(1)).norm(dim=0) ** 2
            # dK[n, c] = scale * sum_m dS[m, n] * Q[m, c]
            dP = gf[bb, hh] @ vf[bb, hh].T
            delta = (p * dP).sum(-1, keepdim=True)
            dS = p * (dP - delta)
            cond_k[bb, hh // rep] += (dS.unsqueeze(-1) * qf[
                bb, hh].unsqueeze(1)).norm(dim=0) ** 2
    return cond_k.sqrt() * scale, cond_v.sqrt()


def _assert_close_grad_tol(got, exp, dtype, cond=None):
    """assert_close with an extra bf16-rounding allowance: cells fed by
    cancelling q-row sums get tol widened by 4*eps*||terms||_2 (the
    observed random-walk error is ~1*eps*l2, so 4x keeps real O(0.1)
    mapping bugs detectable).  cond=None keeps the plain tolerance."""
    if cond is None or dtype != BF16:
        torch.testing.assert_close(got, exp, **TOL[dtype])
        return
    atol, rtol = TOL[dtype]["atol"], TOL[dtype]["rtol"]
    diff = (got.double() - exp.double()).abs()
    tol = atol + rtol * exp.double().abs() + 4 * (2 ** -8) * cond
    bad = diff > tol
    nbad = int(bad.sum().item())
    assert nbad == 0, (
        f"{nbad}/{diff.numel()} grad cells exceed cancellation-aware "
        f"tolerance; worst diff {diff[bad].max().item():.6g}")


class TestFuzz:
    def test_random_shapes_and_masks(self):
        # Randomized coverage of the accept path: shapes / GQA ratios /
        # headdims / mask geometries a serving stack could plausibly send.
        g = torch.Generator(device="cpu").manual_seed(20260826)
        torch.manual_seed(20260826)
        for i in range(30):
            hkv = torch.randint(1, 3, (1,), generator=g).item()
            hq = hkv * torch.randint(1, 5, (1,), generator=g).item()
            b = torch.randint(1, 3, (1,), generator=g).item()
            sq = torch.randint(16, 1025, (1,), generator=g).item()
            sk = torch.randint(16, 1025, (1,), generator=g).item()
            d = torch.tensor([16, 64, 96, 128, 192, 256])[
                torch.randint(0, 6, (1,), generator=g)].item()
            q, k, v = _mk(b, hq, hkv, sq, sk, d)
            if i % 3 == 0:
                mask = _rand_stair_mask(sq, sk, seed=i)
            elif i % 3 == 1:
                mask = _rand_band_mask(sq, sk, seed=i)
            else:
                mask = _causal_mask(sq, sk)
            print(f"FUZZ_CONFIG i={i} b={b} hq={hq} hkv={hkv} sq={sq} "
                  f"sk={sk} d={d} mask_type={i % 3}")
            try:
                # fwd on the shared helper, grads checked with the
                # cancellation-aware tolerance (bf16 sums over q rows can
                # cancel to ~0 while per-term rounding stays ~eps*l2).
                dtype = q.dtype
                out = _run_arb(q, k, v, mask=mask)
                ref = _ref_math(q, k, v, mask=mask)
                torch.testing.assert_close(out, ref, **TOL[dtype])
                gout = torch.randn_like(out)
                _assert_close_arb.last_grad = gout
                out.backward(gout)
                exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, gout,
                                                      mask=mask)
                cond_k, cond_v = _cancel_cond_grads(q, k, v, gout, mask)
                torch.testing.assert_close(q.grad, exp_q, **TOL[dtype])
                _assert_close_grad_tol(k.grad, exp_k, dtype, cond_k)
                _assert_close_grad_tol(v.grad, exp_v, dtype, cond_v)
            except Exception:
                print(f"FUZZ_FAIL i={i} b={b} hq={hq} hkv={hkv} sq={sq} "
                      f"sk={sk} d={d} mask_type={i % 3}")
                torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(),
                            "v": v.detach().cpu(),
                            "mask": mask.detach().cpu(),
                            "g": getattr(_assert_close_arb, "last_grad",
                                         torch.empty(0)).detach().cpu()},
                           "/tmp/fuzz_fail.pt")
                print("FUZZ_FAIL_DUMP /tmp/fuzz_fail.pt")
                raise


# --------------------------------------------------------------------------
# O. CUDA graph behavior (host-sync probe + generator interaction)
# --------------------------------------------------------------------------
# FINDINGS (pinned by the isolated probes below):
#   * capture is UNSUPPORTED: the decompose probe does .item() host syncs
#     which invalidate stream capture -> clean RuntimeError.
#   * a failed capture attempt PERMANENTLY poisons the default CUDA
#     generator ("Offset increment outside graph capture encountered
#     unexpectedly" on every later randn; re-seeding does NOT recover).
#     This PPU-fork behavior makes an in-process capture test fatal to
#     the rest of the suite, so the probes run ISOLATED in subprocesses
#     and are OPT-IN (SDPA_RUN_GRAPH_CAPTURE=1) to keep the default CI
#     run deterministic and side-effect free.
# Production guidance: never capture this backend inside CUDA graphs.
class TestCudaGraph:
    _PROBE_CAPTURE = r"""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
mask = torch.ones(128, 128, device="cuda", dtype=torch.bool)
def run():
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
    run(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            run()
        print("CAPTURE:OK")
    except RuntimeError:
        print("CAPTURE:CLEAN_ERROR")
try:
    torch.randn(2, device="cuda")
    print("RNG:OK")
except RuntimeError:
    print("RNG:POISONED")
    torch.cuda.manual_seed_all(0)
    try:
        torch.randn(2, device="cuda")
        print("RESEED:RECOVERED")
    except RuntimeError:
        print("RESEED:UNRECOVERABLE")
"""

    @staticmethod
    def _run_isolated():
        import os
        import subprocess
        import sys
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        r = subprocess.run([sys.executable, "-c",
                            TestCudaGraph._PROBE_CAPTURE],
                           capture_output=True, text=True, timeout=300,
                           env=env)
        assert r.returncode == 0, \
            f"probe crashed (rc={r.returncode}): {r.stderr[-800:]}"
        return r.stdout

    @pytest.mark.skipif(
        not os.environ.get("SDPA_RUN_GRAPH_CAPTURE"),
        reason="graph-capture probes poison the default CUDA generator; "
               "opt in with SDPA_RUN_GRAPH_CAPTURE=1")
    def test_capture_unsupported_and_rng_poisoning(self):
        out = self._run_isolated()
        assert "CAPTURE:CLEAN_ERROR" in out, \
            f"capture behavior changed: {out}"
        assert "RNG:POISONED" in out, \
            f"generator poisoning behavior changed: {out}"
        assert "RESEED:UNRECOVERABLE" in out, \
            f"reseed recovery behavior changed: {out}"


# --------------------------------------------------------------------------
# P. grad-mask combinations & unusual tensor layouts
# --------------------------------------------------------------------------
class TestGradAndLayouts:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7])
    def test_grad_input_combinations(self, bits):
        # every non-empty subset of {q, k, v} requiring grad
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        q.requires_grad_(bool(bits & 1))
        k.requires_grad_(bool(bits & 2))
        v.requires_grad_(bool(bits & 4))
        out = _run_arb(q, k, v, mask=mask)
        g = torch.randn_like(out)
        out.backward(g)
        exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, g, mask=mask)
        for got, exp, on in ((q.grad, exp_q, bits & 1),
                             (k.grad, exp_k, bits & 2),
                             (v.grad, exp_v, bits & 4)):
            if not on:
                assert got is None
            else:
                torch.testing.assert_close(got, exp, **TOL[BF16])

    def test_packed_qkv_non_contiguous(self):
        # production layout: q/k/v sliced out of one packed buffer.
        # Views are non-leaf, so grads are pulled with autograd.grad.
        b, s, h, d = 1, 256, 2, 64
        packed = torch.randn(b, s, 3 * h * d, device=DEVICE, dtype=BF16,
                             requires_grad=True)
        qs, ks, vs = packed.chunk(3, dim=-1)
        q = qs.view(b, s, h, d).permute(0, 2, 1, 3)
        k = ks.view(b, s, h, d).permute(0, 2, 1, 3)
        v = vs.view(b, s, h, d).permute(0, 2, 1, 3)
        assert not q.is_contiguous()
        mask = _causal_mask(s, s)
        out = _run_arb(q, k, v, mask=mask)
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])
        g = torch.randn_like(out)
        (got,) = torch.autograd.grad(out, packed, g)
        qc = q.detach().contiguous().requires_grad_(True)
        kc = k.detach().contiguous().requires_grad_(True)
        vc = v.detach().contiguous().requires_grad_(True)
        ref2 = _ref_math(qc, kc, vc, mask=mask, detach_inputs=False)
        ref2.backward(g)
        exp = torch.cat([
            qc.grad.permute(0, 2, 1, 3).reshape(b, s, h * d),
            kc.grad.permute(0, 2, 1, 3).reshape(b, s, h * d),
            vc.grad.permute(0, 2, 1, 3).reshape(b, s, h * d)], dim=-1)
        torch.testing.assert_close(got, exp, **TOL[BF16])

    def test_broadcast_grad_out(self):
        # grads arriving as stride-0 expanded views (common downstream of
        # parameter sharing / broadcast losses)
        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        mask = _causal_mask(256, 256)
        out = _run_arb(q, k, v, mask=mask)
        g_narrow = torch.randn(2, 1, 256, 64, device=DEVICE, dtype=BF16)
        out.backward(g_narrow.expand_as(out))
        exp_q, exp_k, exp_v = _ref_math_grads(
            q, k, v, g_narrow.expand_as(out), mask=mask)
        for got, exp in ((q.grad, exp_q), (k.grad, exp_k), (v.grad, exp_v)):
            torch.testing.assert_close(got, exp, **TOL[BF16])

    def test_non_contiguous_mask(self):
        base = _causal_mask(256, 256)
        mask = base.transpose(0, 1)  # transposed view, still bool 2-D
        assert not mask.is_contiguous()
        q, k, v = _mk(1, 2, 2, 256, 256, 64)
        _assert_close_arb(q, k, v, mask=mask)


# --------------------------------------------------------------------------
# Q. autocast (AMP)
# --------------------------------------------------------------------------
class TestAutocast:
    def test_autocast_bf16_routes_and_matches(self):
        # fp32 inputs under autocast: sdpa casts to bf16 before backend
        # selection, so the flex flash attention backend must accept and be correct.
        q = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=FP32)
        k = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=FP32)
        v = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=FP32)
        mask = _causal_mask(256, 256)
        with arb_only(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.dtype == BF16
        ref = _ref_math(q.bfloat16(), k.bfloat16(), v.bfloat16(), mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])


# --------------------------------------------------------------------------
# R. inference_mode / no_grad
# --------------------------------------------------------------------------
class TestInferenceModes:
    def test_inference_mode(self):
        with torch.inference_mode():
            q = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
            k = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
            v = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
            mask = _causal_mask(256, 256)
            with arb_only():
                out = F.scaled_dot_product_attention(q, k, v,
                                                     attn_mask=mask)
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])

    def test_no_grad(self):
        q = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
        v = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
        mask = _causal_mask(256, 256)
        with torch.no_grad():
            out = _run_arb(q, k, v, mask=mask)
        assert out.grad_fn is None
        torch.testing.assert_close(out, _ref_math(q, k, v, mask=mask),
                                   **TOL[BF16])


# --------------------------------------------------------------------------
# S. Cache-eviction soak (resolved/decompose caches under churn)
# --------------------------------------------------------------------------
class TestCacheSoak:
    def test_many_masks_bounded_memory(self):
        # 48 distinct mask identities churn both caches past their
        # eviction thresholds; results must stay correct and memory
        # bounded.
        torch.manual_seed(7)
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        for i in range(2):  # warm the allocator
            _run_arb(q, k, v, mask=_rand_stair_mask(256, 256, i))
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        for i in range(48):
            s = 128 + (i % 16) * 16
            qi, ki, vi = _mk(1, 2, 2, s, s, 64, requires_grad=False)
            mask = _rand_stair_mask(s, s, seed=1000 + i)
            out = _run_arb(qi, ki, vi, mask=mask)
            assert torch.isfinite(out).all()
            if i in (17, 33, 47):  # post-eviction correctness spot checks
                torch.testing.assert_close(
                    out, _ref_math(qi, ki, vi, mask=mask), **TOL[BF16])
        torch.cuda.empty_cache()
        delta = torch.cuda.memory_allocated() - base_mem
        assert delta < 64 * 1024 * 1024, f"cache leak: {delta} bytes"


# --------------------------------------------------------------------------
# T. Training-loop integration (end-to-end through a real module)
# --------------------------------------------------------------------------
class TestTrainingIntegration:
    def test_loss_decreases_over_steps(self):
        torch.manual_seed(123)
        h, d, s, b = 4, 64, 128, 2

        model = TinyAttn(nhead=h, dim=d).to(DEVICE)
        x = torch.randn(b, s, d, device=DEVICE)
        target = torch.randn(b, s, d, device=DEVICE)
        mask = _causal_mask(s, s)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            with arb_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(x, mask), target)
            loss.backward()
            assert all(torch.isfinite(p.grad).all()
                       for p in model.parameters())
            opt.step()
            losses.append(loss.item())
            assert torch.isfinite(loss)
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# --------------------------------------------------------------------------
# U. Double backward
# --------------------------------------------------------------------------
class TestDoubleBackward:
    def test_double_backward_clean_behavior(self):
        # Double backward is not a shipped capability; the contract is
        # CLEAN failure (RuntimeError), never a crash or wrong grads.
        q, k, v = _mk(1, 2, 2, 128, 128, 64)
        mask = _causal_mask(128, 128)
        with arb_only():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            try:
                g1, = torch.autograd.grad(out.sum(), q, create_graph=True)
                torch.autograd.grad(g1.sum(), q)
                double_supported = True
            except RuntimeError:
                double_supported = False
        # either way the library must stay usable
        out_after = _run_arb(q.detach(), k.detach(), v.detach(), mask=mask)
        assert torch.isfinite(out_after).all()
        TestDoubleBackward.supported = double_supported  # documented


# --------------------------------------------------------------------------
# V. Long-sequence numerics
# --------------------------------------------------------------------------
class TestLongSequence:
    def test_8k_fwd_bwd(self):
        q, k, v = _mk(1, 2, 2, 8192, 8192, 64)
        mask = _causal_mask(8192, 8192)
        _assert_close_arb(q, k, v, mask=mask)

    def test_16k_fwd(self):
        q, k, v = _mk(1, 1, 1, 16384, 16384, 64, requires_grad=False)
        mask = _causal_mask(16384, 16384)
        out = _run_arb(q, k, v, mask=mask)
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, atol=4e-2, rtol=2e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

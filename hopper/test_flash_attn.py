import os
import math
import itertools

import pytest
import torch
import torch.nn.functional as F
from torch._C import parse_schema

from einops import rearrange, repeat
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    apply_rotary_emb = None

from padding import pad_input, unpad_input
from test_util import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)

from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_combine
from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata

cuda_cap = 80
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    cuda_cap = major * 10 + minor

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", ("TRUE" if cuda_cap < 89 else "FALSE")) == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_HDIMV = os.getenv("FLASH_ATTENTION_DISABLE_HDIMV", "FALSE") == "TRUE"

COMPILED_HDIMS = (
    []
    + ([64] if not DISABLE_HDIM64 else [])
    + ([96] if not DISABLE_HDIM96 else [])
    + ([128] if not DISABLE_HDIM128 else [])
    + ([192] if not DISABLE_HDIM192 else [])
    + ([256] if not DISABLE_HDIM256 else [])
)


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("V_colmajor", [False, True])
@pytest.mark.parametrize("V_colmajor", [False])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize("d", [64, 128, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128, 192])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (64, 128),
        (128, 192),
        (256, 256),
        (239, 1),
        (799, 3),
        (113, 203),
        (113, 128),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (384, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (4096, 4096),
        (4224, 4224),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(128, 128)])
def test_flash_attn_output(
        seqlen_q, seqlen_k, d, causal, local, softcap, V_colmajor, deterministic, has_qv, mha_type, dtype
):
    if V_colmajor and (seqlen_k % 16 != 0 or dtype != torch.float8_e4m3fn):
        pytest.skip("V_colmajor requires seqlen_k to be a multiple of 16 and dtype to be float8_e4m3fn")
    if d == 96 and dtype == torch.float8_e4m3fn:
        pytest.skip("head dim 96 not support for dtype float8_e4m3fn now")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_k <= 2048 else 2
    nheads = 6
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if DISABLE_HDIMV or dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4)
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,)).tolist()
        # window_size = (-1, -1) if not local else (16, 0)
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().to(dtype).requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach().to(dtype).requires_grad_() if has_qv else None
        if V_colmajor:
            v = rearrange(rearrange(v.detach(), "b s h d -> b h d s").contiguous(), "b h d s -> b s h d").requires_grad_()
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            None,
            None,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )

        # qk = torch.einsum('bshd,bthd->bhst', q_ref, k_ref).float()
        # if qv is not None:
        #     qk += torch.einsum('bshd,bthd->bhst', qv_ref, v_ref).float()
        # m = qk.amax(-1, keepdim=True)
        # s_tmp = torch.exp((qk - m) / math.sqrt(d))
        # exp_sum = s_tmp.sum(-1)
        # qk = torch.einsum('bthd,bshd->bhts', q_ref.float() / math.sqrt(d), k_ref.float())
        # lse_ref = torch.logsumexp(qk, dim=-1)

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                qv=qv,
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                pack_gqa=pack_gqa,
                num_splits=num_splits
            )
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most twice the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol

        if (
            not DISABLE_BACKWARD
            and dtype != torch.float8_e4m3fn
            and not V_colmajor
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
        ):
            g = torch.randn_like(out)
            do_o = ((g.float() * out.float()).sum(-1)).transpose(1, 2)
            # import flash_attn_3_cuda
            # dq, dk, dv, softmax_d, dq_accum, dk_accum, dv_accum = flash_attn_3_cuda.bwd(
            #     g,
            #     q,
            #     k,
            #     v,
            #     out,
            #     lse,
            #     None,
            #     None,
            #     None,
            #     d ** (-0.5),
            #     causal,
            #     window_size[0], window_size[1],
            #     softcap,
            #     deterministic,
            #     0,  # sm_margin
            # )
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0

            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - do_o.transpose(1, 2).unsqueeze(1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())

            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
            print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
            print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
            print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
            print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
            print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
            print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dq - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dk - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dv - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float16] if not DISABLE_FP16 else []) + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("has_qv", [False, True])
@pytest.mark.parametrize("has_qv", [False])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
# @pytest.mark.parametrize("softcap", [0.0])
@pytest.mark.parametrize("local", [False] + ([True] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("add_unused_qkv", [False, True])
# @pytest.mark.parametrize("add_unused_qkv", [True])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [64, 96, 128])
@pytest.mark.parametrize("d", COMPILED_HDIMS)
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 3),
        (2, 1),
        (511, 1),
        (3, 513),
        (64, 128),
        (128, 128),
        (256, 256),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (307, 256),
        (640, 128),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_varlen_output(
        seqlen_q, seqlen_k, d, add_unused_qkv, causal, local, softcap, deterministic, has_qv, mha_type, dtype
):
    if d == 96 and dtype == torch.float8_e4m3fn:
        pytest.skip("head dim 96 not support for dtype float8_e4m3fn now")
    device = "cuda"
    # set seed
    torch.random.manual_seed(seqlen_q + seqlen_k + d + int(causal) * 2 + int(local))
    # batch_size = 40
    # nheads = 16
    batch_size = 9 if seqlen_q <= 2048 else 2
    nheads = 6
    # batch_size = 2
    # nheads = 1
    nheads_kv = nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if DISABLE_HDIMV or dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if seqlen_q <= seqlen_k and not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
        if softcap > 0.0:
            # Ensure the values of qk are at least within softcap range.
            q_ref = (q_ref * softcap / 4).detach().requires_grad_()
        q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
        k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
        if has_qv:
            qv_ref = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv_ref = None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
        if dtype == torch.float8_e4m3fn:
            q_descale, k_descale, v_descale = [torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2 for _ in range(3)]
        else:
            q_descale, k_descale, v_descale = None, None, None
        q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
        qv = qv_ref.detach() if has_qv else None
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random", zero_lengths=False
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, device, mode="random", zero_lengths=True
        )

        def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
            if add_unused:
                another_mask = generate_random_padding_mask(max_seq_len, bs, device)
                attn_mask = torch.logical_and(padding_mask, another_mask)
                unused_mask = torch.logical_xor(
                    torch.logical_or(padding_mask, another_mask), attn_mask
                )
            else:
                attn_mask = padding_mask
                unused_mask = None
            return attn_mask, unused_mask

        query_padding_mask, query_unused_mask = _gen_unused_masks(
            query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device
        )
        key_padding_mask, key_unused_mask = _gen_unused_masks(
            key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
        )

        (
            q_unpad,
            k_unpad,
            v_unpad,
            qv_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            qv,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, qv=qv, kvpacked=False,
                        query_unused_mask=query_unused_mask, key_unused_mask=key_unused_mask)
        q_unpad, k_unpad, v_unpad = [x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)]
        out_ref, attn_ref = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap
        )
        out_pt, attn_pt = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv_ref,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            upcast=False,
            reorder_ops=True,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        )


        print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

        if query_unused_mask is not None:
            q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

        # Numerical error if we just do any arithmetic on out_ref
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3

        pack_gqa_vals = [False, True] if not DISABLE_PACKGQA else [False]
        num_splits_vals = [1, 3] if not DISABLE_SPLIT else [1]
        for pack_gqa, num_splits in itertools.product(pack_gqa_vals, num_splits_vals):
            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                causal=causal,
                qv=qv_unpad,
                q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
            )
            out = output_pad_fn(out_unpad)
            if query_unused_mask is not None:
                out.masked_fill_(q_zero_masking, 0.0)
            print(f"Output max diff: {(out - out_ref).abs().max().item()}")
            print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
            # if not causal:
            #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
            # breakpoint()

            # Check that FlashAttention's numerical error is at most 3x the numerical error
            # of a Pytorch implementation.
            assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol


        if (
            not DISABLE_BACKWARD
            and dtype != torch.float8_e4m3fn
            and not has_qv
            and not dv > 256
            and not attention_chunk != 0
        ):
            g_unpad = torch.randn_like(out_unpad)
            do_o = ((g_unpad.float() * out_unpad.float()).sum(-1)).transpose(-1, -2)
            # import flash_attn_3_cuda
            # dq_unpad, dk_unpad, dv_unpad, softmax_d, dq_accum, lse_log2 = flash_attn_3_cuda.bwd_varlen(
            #     g_unpad,
            #     q_unpad,
            #     k_unpad,
            #     v_unpad,
            #     out_unpad,
            #     lse,
            #     None,
            #     None,
            #     None,
            #     cu_seqlens_q,
            #     cu_seqlens_k,
            #     None, None,
            #     max_seqlen_q,
            #     max_seqlen_k,
            #     d ** (-0.5),
            #     causal,
            #     window_size[0], window_size[1],
            #     softcap,
            #     deterministic,
            #     0,  # sm_margin
            # )
            dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad)
            dq = dq_pad_fn(dq_unpad)
            dk = dk_pad_fn(dk_unpad)
            dv = dk_pad_fn(dv_unpad)
            if key_unused_mask is not None:
                k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
                dk.masked_fill_(k_zero_masking, 0.0)
                dv.masked_fill_(k_zero_masking, 0.0)
            if query_unused_mask is not None:
                dq.masked_fill_(q_zero_masking, 0.0)
            # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
            # assert (softmax_d - do_o).abs().max().item() <= 1e-5
            # assert dq_accum.abs().max().item() == 0.0
            g = output_pad_fn(g_unpad)

            # qk = torch.einsum('bthd,bshd->bhts', q / (d ** 0.5), k).float()
            # qk = torch.masked_fill(qk, rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
            # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
            # P = torch.softmax(qk, -1)
            # dP = P * (dS - (g.float() * out.float()).sum(-1).transpose(1, 2).unsqueeze(-1))
            # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
            # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
            # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())


            # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
            dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
            print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
            print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
            print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
            print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
            print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
            print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
            print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
            print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
            print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
            print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
            print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
            print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
            # breakpoint()
            dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dq - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
            dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dk - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
            dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
            assert (dv - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.bfloat16] + ([torch.float8_e4m3fn] if not DISABLE_FP8 else []))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize("mha_type", ["mha"])
@pytest.mark.parametrize("new_kv", [False] + ([True] if not DISABLE_APPENDKV else []))
# @pytest.mark.parametrize("new_kv", [True])
@pytest.mark.parametrize("causal,local", [(False, False), (True, False)] + ([(False, True)] if not DISABLE_LOCAL else []))
# @pytest.mark.parametrize("causal,local", [(False, False), (True, False)])
# @pytest.mark.parametrize("causal,local", [(False, False)])
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True, False] if not DISABLE_APPENDKV else [True])
# @pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [True])
@pytest.mark.parametrize("has_rotary_seqlens", [False, True])
# @pytest.mark.parametrize("has_rotary_seqlens", [False])
@pytest.mark.parametrize("rotary_interleaved", [False, True] if not DISABLE_APPENDKV else [False])
# @pytest.mark.parametrize("rotary_interleaved", [True])
@pytest.mark.parametrize("rotary_fraction", [0.0, 0.5, 1.0] if (not DISABLE_APPENDKV) and (apply_rotary_emb is not None) else [0.0])
# @pytest.mark.parametrize("rotary_fraction", [0.0])
@pytest.mark.parametrize("page_size", [None] + ([1, 16, 64] if not DISABLE_PAGEDKV else []))
# @pytest.mark.parametrize("page_size", [None])
@pytest.mark.parametrize("has_leftpad", [False, True])
# @pytest.mark.parametrize("has_leftpad", [False])
@pytest.mark.parametrize("has_batch_idx", [False, True])
# @pytest.mark.parametrize("has_batch_idx", [False])
@pytest.mark.parametrize("varlen_q", [False, True])
# @pytest.mark.parametrize("varlen_q", [False])
# @pytest.mark.parametrize("d", [32, 59, 64, 80, 128, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [128, 192])
# @pytest.mark.parametrize("d", [192])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        (64, 256),
        (3, 799),
        (64, 2048),
        (16, 20000),
        # (1, 128 * 1024),
        # (16, 128 * 1024),
        (128, 128),
        (256, 512),  # To test appending KV with more than 1 block
        (2048, 3577),  # Enough tile to test persistent scheduler
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    varlen_q,
    has_batch_idx,
    has_leftpad,
    page_size,
    rotary_fraction,
    rotary_interleaved,
    has_rotary_seqlens,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    mha_type,
    dtype,
):
    if page_size is not None and seqlen_k % page_size != 0:
        pytest.skip()
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv and rotary_fraction > 0.0:
        pytest.skip()
    if rotary_fraction == 0.0 and has_rotary_seqlens:
        pytest.skip()
    if dtype == torch.float8_e4m3fn:
        if cuda_cap < 89:
            pytest.skip()
        if d == 96:
            pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    # batch_size = 1
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    nheads = 6
    # nheads = 1
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    dv_vals = [128, d] if d > 128 and d <= 192 else ([256, 512, d] if d <= 64 else [d])
    if DISABLE_HDIMV or dtype == torch.float8_e4m3fn:
        dv_vals = [d]
    attention_chunk_vals = [torch.randint(1, seqlen_k * 2, (1,)).item(), 0] if (causal or local) and not DISABLE_LOCAL else [0]
    for dv, attention_chunk in itertools.product(dv_vals, attention_chunk_vals):
        # has_qv = d == 64 and dv >= 256
        has_qv = False   # for sm80
        q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        if has_qv:
            qv = torch.randn(batch_size, seqlen_q, nheads, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
        else:
            qv = None
        if varlen_q:
            query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(q, query_padding_mask)
            output_pad_fn = lambda output_unpad: pad_input(
                output_unpad, indices_q, batch_size, seqlen_q
            )
            qv_unpad = rearrange(qv, "b s ... -> (b s) ...")[indices_q] if has_qv else None
        else:
            query_padding_mask = None
            q_unpad = q
            qv_unpad = qv
            cu_seqlens_q, max_seqlen_q = None, None
        # Put window_size after QKV randn so that window_size changes from test to test
        window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

        seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
        cu_seqlens_k_new = None
        key_new_padding_mask = None
        if new_kv:
            k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            v = torch.randn(batch_size, seqlen_new, nheads_k, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            if varlen_q:  # k & v are also varlen
                key_new_padding_mask = generate_random_padding_mask(seqlen_new, batch_size, device, mode="random")
                k_unpad, indices_k, cu_seqlens_k_new, *rest = unpad_input(k, key_new_padding_mask)
                v_unpad, *rest = unpad_input(v, key_new_padding_mask)
            else:
                k_unpad, v_unpad = k, v
        else:
            k, v, k_unpad, v_unpad = None, None, None, None
        if page_size is None:
            k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref)
            page_table = None
        else:
            (
                k_cache,
                v_cache,
                page_table,
                k_cache_paged,
                v_cache_paged,
                num_blocks,
            ) = _generate_block_kvcache(
                seqlen_k, page_size, batch_size_cache, nheads_k, d, dv, device, dtype, dtype_ref
            )
        cache_seqlens = torch.randint(
            0 if new_kv else 1,
            # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
            (
                (seqlen_k - (seqlen_q if (causal or local) and rotary_dim > 1 else seqlen_new) + 1)
                if new_kv
                else (seqlen_k + 1)
            ),
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        if has_leftpad:
            cache_leftpad = torch.cat([torch.randint(0, cache_seqlens[i].item(), (1,), dtype=torch.int32, device=device)
                                    if cache_seqlens[i].item() > 0 else torch.zeros(1, dtype=torch.int32, device=device)
                                    for i in range(batch_size)])
        else:
            cache_leftpad = None
        if has_batch_idx:
            cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
                :batch_size
            ]
        else:
            cache_batch_idx = None
        arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
        cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
        if not new_kv:
            key_padding_mask = arange < cache_seqlens_expanded
        else:
            k_new_seqlens = key_new_padding_mask.sum(-1, keepdims=True) if varlen_q else seqlen_new
            key_padding_mask = arange < cache_seqlens_expanded + k_new_seqlens
        if has_leftpad:
            key_padding_mask = torch.logical_and(
                key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
            )
        # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
        rotary_seqlens = cache_seqlens if not has_rotary_seqlens else cache_seqlens // 2
        if rotary_dim > 0:
            angle = (
                torch.rand(
                    seqlen_k if page_size is None else num_blocks * page_size,
                    rotary_dim // 2,
                    device=device,
                )
                * 2
                * math.pi
            )
            cos = torch.cos(angle).to(dtype=dtype_ref).to(dtype).to(dtype_ref)
            sin = torch.sin(angle).to(dtype=dtype_ref).to(dtype).to(dtype_ref)
            if causal or local:
                q_ro = apply_rotary_emb(
                    q, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved
                )
            else:
                q_ro = rearrange(
                    apply_rotary_emb(
                        rearrange(q, "b s h d -> b 1 (s h) d"),
                        cos,
                        sin,
                        seqlen_offsets=rotary_seqlens,
                        interleaved=rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=seqlen_q,
                )
            # q_ro = q
            k_ro = apply_rotary_emb(
                k, cos, sin, seqlen_offsets=rotary_seqlens, interleaved=rotary_interleaved
            )
        else:
            cos, sin = None, None
            q_ro, k_ro = q, k
        # k_cache[:, 64:] = -1
        k_cache_ref = (k_cache if not has_batch_idx else k_cache[cache_batch_idx]).clone()
        v_cache_ref = (v_cache if not has_batch_idx else v_cache[cache_batch_idx]).clone()
        if new_kv:
            update_mask = torch.logical_and(
                cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + k_new_seqlens
            )
            k_to_update = rearrange(k_ro, "b s ... -> (b s) ...")
            v_to_update = rearrange(v, "b s ... -> (b s) ...")
            if varlen_q:
                k_to_update = k_to_update[indices_k]
                v_to_update = v_to_update[indices_k]
            k_cache_ref[update_mask] = k_to_update
            v_cache_ref[update_mask] = v_to_update
        k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        out_ref, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            attention_chunk=attention_chunk,
            key_leftpad=cache_leftpad,
        )
        out_pt, _ = attention_ref(
            q_ro,
            k_cache_rep,
            v_cache_rep,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            qv=qv,
            window_size=window_size,
            attention_chunk=attention_chunk,
            upcast=False,
            reorder_ops=True,
            key_leftpad=cache_leftpad,
            intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None
        )
        q = q.to(dtype)
        q_unpad = q_unpad.to(dtype) if varlen_q else None
        k_cache = k_cache.to(dtype)
        v_cache = v_cache.to(dtype)
        k_cache_paged = k_cache_paged.to(dtype) if page_size is not None else None
        v_cache_paged = v_cache_paged.to(dtype) if page_size is not None else None
        k = k.to(dtype) if k is not None else None
        v = v.to(dtype) if v is not None else None
        k_unpad = k_unpad.to(dtype) if k_unpad is not None else None
        v_unpad = v_unpad.to(dtype) if v_unpad is not None else None
        qv = qv.to(dtype) if qv is not None else None
        qv_unpad = qv_unpad.to(dtype) if (varlen_q and qv is not None) else None
        cos = cos.to(dtype) if cos is not None else None
        sin = sin.to(dtype) if sin is not None else None
        k_cache_saved = k_cache.clone() if page_size is None else k_cache_paged.clone()
        v_cache_saved = v_cache.clone() if page_size is None else v_cache_paged.clone()
        num_splits_vals = [1, 0] if not DISABLE_SPLIT else [1]
        precompute_metadata_vals = [False, True]
        for num_splits, precompute_metadata in itertools.product(num_splits_vals, precompute_metadata_vals):
            if precompute_metadata:
                scheduler_metadata = get_scheduler_metadata(
                    batch_size, max_seqlen_q if varlen_q else seqlen_q, seqlen_k, nheads, nheads_k, d,
                    cache_seqlens, q.dtype, headdim_v=dv, cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new, cache_leftpad=cache_leftpad,
                    max_seqlen_k_new=seqlen_new, page_size=page_size,
                    causal=causal, window_size=window_size, attention_chunk=attention_chunk,
                    num_splits=num_splits
                )
            else:
                scheduler_metadata = None
            # Repeat to test metadata reuse
            for _ in range(1 if not precompute_metadata else 2):
                if page_size is None:
                    k_cache.copy_(k_cache_saved)
                    v_cache.copy_(v_cache_saved)
                else:
                    k_cache_paged.copy_(k_cache_saved)
                    v_cache_paged.copy_(v_cache_saved)
                out, lse, *rest = flash_attn_with_kvcache(
                    q if not varlen_q else q_unpad,
                    k_cache if page_size is None else k_cache_paged,
                    v_cache if page_size is None else v_cache_paged,
                    k if not new_kv or not varlen_q else k_unpad,
                    v if not new_kv or not varlen_q else v_unpad,
                    qv=qv if not varlen_q else qv_unpad,
                    rotary_cos=cos,
                    rotary_sin=sin,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=cache_batch_idx,
                    cache_leftpad=cache_leftpad,
                    page_table=page_table,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=max_seqlen_q,
                    rotary_seqlens=rotary_seqlens,
                    causal=causal,
                    window_size=window_size,
                    attention_chunk=attention_chunk,
                    rotary_interleaved=rotary_interleaved,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    return_softmax_lse=True
                )
                if varlen_q:
                    out = output_pad_fn(out)
                # out = flash_attn_with_kvcache(
                #     q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=causal, window_size=window_size
                # )
                # out = flash_attn_with_kvcache(q, k_cache, v_cache, causal=causal, window_size=window_size)
                # qk = torch.einsum("bqhd,bkhd->bhqk", q, k_cache_ref)
                # m = qk.amax(-1, keepdim=True)
                # s_tmp = torch.exp((qk - m) / math.sqrt(d))
                # o1 = torch.einsum('bhst,bthd->bshd', s_tmp, v_cache_ref)
                # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
                # probs = torch.softmax(qk, dim=-1)
                print(f"Output max diff: {(out - out_ref).abs().max().item()}")
                print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
                print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
                print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
                # breakpoint()

                # Check that FlashAttention's numerical error is at most twice the numerical error
                # of a Pytorch implementation.
                if new_kv:
                    if page_size is None:
                        k_cache_select = (
                            k_cache.to(dtype_ref) if not has_batch_idx else k_cache.to(dtype_ref)[cache_batch_idx]
                        )
                        v_cache_select = (
                            v_cache.to(dtype_ref) if not has_batch_idx else v_cache.to(dtype_ref)[cache_batch_idx]
                        )
                    else:
                        k_cache_select = rearrange(
                            k_cache_paged.to(dtype_ref)[(page_table if not has_batch_idx else page_table[cache_batch_idx]).flatten()],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                        v_cache_select = rearrange(
                            v_cache_paged.to(dtype_ref)[(page_table if not has_batch_idx else page_table[cache_batch_idx]).flatten()],
                            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                            b=batch_size,
                        )[:, :seqlen_k].to(dtype_ref)
                    k_cache_ref = k_cache_ref.to(dtype).to(dtype_ref)
                    v_cache_ref = v_cache_ref.to(dtype).to(dtype_ref)
                    if dtype is not torch.float8_e4m3fn:
                        assert torch.equal(v_cache_select, v_cache_ref)
                    else:
                        assert torch.allclose(v_cache_select, v_cache_ref, rtol=1e-3, atol=1e-3)
                    # breakpoint()
                    # if rotary_dim == 0 and dtype is not torch.float8_e4m3fn:
                    if rotary_dim == 0:
                        assert torch.equal(k_cache_select, k_cache_ref)
                    else:
                        # if not torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3):
                        #     breakpoint()
                        if dtype is not torch.float8_e4m3fn:
                            assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
                        else:
                            assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-1, atol=1e-1)
                mult = 4 if dtype == torch.float8_e4m3fn else 2
                assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5
                mult_mean = 3 if dtype == torch.float8_e4m3fn else 1.5
                assert (out - out_ref).abs().mean().item() <= mult_mean * (out_pt - out_ref).abs().mean().item()


def _generate_block_kvcache(seqlen_k, page_size, batch_size, nheads_k, d, dv, device, dtype, dtype_ref):
    num_blocks = math.ceil(seqlen_k / page_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, d, device=device, dtype=dtype_ref
    ).to(dtype).to(dtype_ref)
    v_cache_paged = torch.randn(
        num_blocks, page_size, nheads_k, dv, device=device, dtype=dtype_ref
    ).to(dtype).to(dtype_ref)
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, num_blocks


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (64, 8192),
    ],
)
def test_flash_attn_cluster(seqlen_q, seqlen_k, d, causal, dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 16
    nheads_kv = 4
    # There was a bug where this would cause "unspecified launch failure" due to Cluster
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    for _ in range(100):
        flash_attn_func(q, k, v, causal=causal)


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128])
# @pytest.mark.parametrize('d', [32, 56, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [80])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (2048, 2048),
    ],
)
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, causal, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # Simulate under memory load
    dummy = torch.empty(70 * 1024 ** 3, dtype=torch.uint8, device=device)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    try:
        q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
        torch.random.manual_seed(42)
        out0 = flash_attn_func(q, k, v, causal=causal)
        g = torch.randn_like(out0)
        dq0, dk0, dv0 = torch.autograd.grad(out0, (q, k, v), g)
        # Numerical error if we just do any arithmetic on dq
        dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

        for i in range(1000):
            torch.random.manual_seed(42)
            out = flash_attn_func(q, k, v, causal=causal)
            assert torch.equal(out, out0)
            # assert torch.equal(lse, lse0)

            dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
            dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
            if not dq_equal:
                print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
                # breakpoint()
            assert torch.equal(dv, dv0)
            assert torch.equal(dk, dk0)
            assert dq_equal
    finally:
        del dummy, q, k, v
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def attention_combine_ref(out_partial, lse_partial):
    """
    out_partial: (num_splits, batch_size, seqlen, nheads, d)
    lse_partial: (num_splits, batch_size, nheads, seqlen)
    """
    lse = torch.logsumexp(lse_partial, dim=0)
    scale = torch.exp(lse_partial - lse)
    scale = torch.where(torch.isinf(scale) | torch.isnan(scale), torch.zeros_like(scale), scale)
    out = (scale.unsqueeze(-1) * out_partial).sum(0)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("d", [64, 96, 128, 192, 256, 512])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [1, 2, 3, 32, 64, 256, 113, 108, 640, 1024])
# @pytest.mark.parametrize("seqlen", [12, 32, 64, 256, 112, 108, 640, 1024, 2048, 8192])
# @pytest.mark.parametrize("seqlen", [15])
@pytest.mark.parametrize("num_splits", [2, 3, 5, 17, 32, 55, 97, 133])   # bug fix: combine kernel does not support num_splits=1
# @pytest.mark.parametrize("num_splits", [1, 2, 3, 5, 11])
# @pytest.mark.parametrize("num_splits", [128])
def test_flash_attn_combine(num_splits, seqlen, d, dtype):
    if DISABLE_SPLIT:
        pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(1)
    batch_size = 5
    nheads = 16
    # batch_size = 1
    # nheads = 1
    out_partial = torch.randn(num_splits * 2, batch_size, nheads, seqlen, d, device=device, dtype=torch.float32).transpose(2, 3)[:num_splits]  # To test non-contiguous tensor
    lse_partial = torch.randn(num_splits, batch_size, nheads * 2, seqlen, device=device, dtype=torch.float32).transpose(-1, -2)[:, :, :, :nheads]  # To test non-contiguous tensor
    # To test short-circuiting based on num_splits
    lse_partial[num_splits // 2:, :batch_size // 3] = -float("inf")
    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype)
    out_ref, lse_ref = attention_combine_ref(out_partial, lse_partial)
    out_pt = out_ref.to(dtype)

    print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    print(f"LSE mean diff: {(lse - lse_ref).abs().mean().item()}")
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # breakpoint()

    assert torch.allclose(lse, lse_ref, atol=1e-5, rtol=1e-5)
    multiple = 2
    assert ((out - out_ref).abs().max().item() <= multiple * (out_pt - out_ref).abs().max().item()) or torch.allclose(out, out_pt, atol=1e-5, rtol=1e-5)

    # from flash_attn.utils.benchmark import pytorch_profiler
    # # pytorch_profiler(torch.sum, lse_partial)
    # pytorch_profiler(flash_attn_combine, out_partial, lse_partial)
    # pytorch_profiler(torch.sum, out_partial)

def test_flash3_bw_compatibility() -> None:
    # Let's try to always stay backward compatible! This will make life easier
    # for downstream libaries, users, and exported models.
    # 1/ Instead of removing arguments, error out if their value is no longer supported
    # 2/ When adding arguments, add them at the end with a default value
    assert torch.ops.flash_attn_3.fwd.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::fwd(Tensor q, Tensor k, Tensor v, Tensor(k_new!)? k_new=None, "
        "Tensor(v_new!)? v_new=None, Tensor? q_v=None, Tensor(out!)? out=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? cu_seqlens_k_new=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
        "int? max_seqlen_q=None, int? max_seqlen_k=None, Tensor? page_table=None, "
        "Tensor? kv_batch_idx=None, Tensor? leftpad_k=None, Tensor? rotary_cos=None, Tensor? rotary_sin=None, "
        "Tensor? seqlens_rotary=None, Tensor? q_descale=None, Tensor? k_descale=None, Tensor? v_descale=None, "
        "float? softmax_scale=None, bool is_causal=False, int window_size_left=-1, int window_size_right=-1, "
        "int attention_chunk=0, float softcap=0., bool is_rotary_interleaved=False, "
        "Tensor? scheduler_metadata=None, int num_splits=0, bool? pack_gqa=None, int sm_margin=0) "
        "-> (Tensor(out!), Tensor, Tensor, Tensor)"
    ))
    assert torch.ops.flash_attn_3.bwd.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, "
        "Tensor(dq!)? dq=None, Tensor(dk!)? dk=None, Tensor(dv!)? dv=None, Tensor? cu_seqlens_q=None, "
        "Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, int? max_seqlen_q=None, "
        "int? max_seqlen_k=None, float? softmax_scale=None, bool is_causal=False, int window_size_left=-1, "
        "int window_size_right=-1, float softcap=0., bool deterministic=False, int sm_margin=0) "
        "-> (Tensor(dq!), Tensor(dk!), Tensor(dv!), Tensor, Tensor, Tensor, Tensor, Tensor)"
    ))
    assert torch.ops.flash_attn_3.fwd_combine.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::fwd_combine(Tensor out_partial, Tensor lse_partial, Tensor(out!)? out=None, "
        "ScalarType? out_dtype=None) -> (Tensor(out!), Tensor)"
    ))
    assert torch.ops.flash_attn_3.get_scheduler_metadata.default._schema.is_backward_compatible_with(parse_schema(
        "flash_attn_3::get_scheduler_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, "
        "int num_heads, int num_heads_k, int headdim, int headdim_v, ScalarType qkv_dtype, Tensor seqused_k, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? cu_seqlens_k_new=None, "
        "Tensor? seqused_q=None, Tensor? leftpad_k=None, int? page_size=None, int max_seqlen_k_new=0, "
        "bool is_causal=False, int window_size_left=-1, int window_size_right=-1, "
        "int attention_chunk=0, bool has_softcap=False, int num_splits=0, bool? pack_gqa=None, "
        "int sm_margin=0) -> Tensor"
    ))


def _mha_wi_paged_kv(k, v, klens):
    """Materialize random physical pages while retaining dense K/V as the reference."""
    heads, page_size = k.shape[1], 64
    pages = sum((n + page_size - 1) // page_size for n in klens) + 1
    kc = torch.empty(pages, page_size, heads, k.shape[-1], device=k.device, dtype=k.dtype)
    vc = torch.empty(pages, page_size, heads, v.shape[-1], device=v.device, dtype=v.dtype)
    table = torch.full((len(klens), (max(klens) + 63) // 64 + 1), -1,
                       device=k.device, dtype=torch.int32)
    lengths = torch.tensor(klens, device=k.device, dtype=torch.int32)
    permutation = torch.randperm(pages, device=k.device)
    def refresh(lengths_cpu, reverse_pages=False):
        table.fill_(-1)
        lengths.copy_(torch.tensor(lengths_cpu, device=k.device, dtype=torch.int32))
        order = permutation.flip(0) if reverse_pages else permutation
        offset = ko = 0
        for batch, n in enumerate(lengths_cpu):
            count = (n + page_size - 1) // page_size
            ids = order[offset:offset + count]
            table[batch, :count] = ids.to(torch.int32)
            for source, cache in [(k, kc), (v, vc)]:
                padded = torch.empty(count * page_size, heads, source.shape[-1],
                                     device=source.device, dtype=source.dtype)
                padded.view(torch.uint8).zero_()
                padded[:n].copy_(source[ko:ko + n])
                cache.view(torch.uint8)[ids] = padded.view(count, page_size, heads, source.shape[-1]).view(torch.uint8)
            ko += n
            offset += count
    refresh(klens)
    return kc, vc, table, lengths, refresh


@pytest.mark.parametrize("heads", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96])
@pytest.mark.parametrize("qlens,klens,causal", [
    ([32], [8192], True),
    ([8192], [8192], True),
    ([8192], [32768], True),
    ([8192], [8192], False),
    ([32, 65], [8192, 16385], True),
    ([32, 0, 33, 1], [8192, 0, 16385, 0], True),
    ([8192, 257, 0], [8192, 4097, 0], True),
    ([8192, 0, 513], [32768, 0, 1025], True),
    ([8192, 257], [8192, 4097], False),
    ([8192, 513], [8192, 0], True),
    ([32] * 8, [8192] * 8, True),
    ([32] * 16, [8192] * 16, True),
    ([8192] * 8, [8192] * 8, True),
    ([4096], [90112], True),
    ([4096] * 2, [90112] * 2, True),
    ([4096], [90112], False),
    ([1024], [16385], True),
    ([4096], [262144], True),
    ([512] * 4, [4096] * 4, True),
    ([64], [16384], True),
    ([32, 32], [131072, 65536], True),
    ([32, 0, 32, 0], [8192, 0, 16385, 0], True),
    ([65, 65], [8193, 8193], True),
    ([1024], [90112], True),
    ([1024] * 3, [90112] * 3, True),
    ([1024, 1023, 1022], [90112, 65535, 0], True),
], ids=["ws", "wi-guarded", "wi-peeled", "wi-noncausal",
        "batch2-ws-ragged", "batch4-ws-empty", "batch3-wi-ragged",
        "batch3-wi-peeled", "batch2-wi-noncausal", "batch2-wi-empty-kv",
        "batch8-ws", "batch16-ws", "batch8-wi", "wi-long-prefix",
        "batch2-wi-long-prefix", "wi-noncausal-long-prefix", "wi-short-query-many-heads",
        "wi-large-kv", "wi-batch4-short-query", "ws-two-query-tiles",
        "ws-long-ragged-kv", "ws-adaptive-splits-empty", "ws-query-and-page-tail",
        "wi-kv-partition", "batch3-wi-kv-partition", "batch3-wi-kv-partition-ragged"])
@pytest.mark.parametrize("layout", ["packed", "paged64", "fixed"])
def test_mha_wi_bf16(heads, qlens, klens, causal, layout):
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("WI/WS prefill requires the qualified device")
    if DISABLE_HDIM192:
        pytest.skip("D192 was disabled at build time")
    fixed, paged = layout == "fixed", layout == "paged64"
    if paged and DISABLE_PAGEDKV:
        pytest.skip("Paged KV was disabled at build time")
    if fixed and (len(set(qlens)) != 1 or len(set(klens)) != 1):
        pytest.skip("Fixed tensors require uniform sequence lengths")
    torch.manual_seed(sum(qlens) + sum(klens) + heads)
    q, k, v = [torch.randn(n, heads, d, device="cuda", dtype=torch.bfloat16)
               for n, d in [(sum(qlens), 192), (sum(klens), 192), (sum(klens), 128)]]
    def cumulative(lengths):
        return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    cq, ck = cumulative(qlens), cumulative(klens)
    from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    if paged:
        kc, vc, table, lengths, refresh = _mha_wi_paged_kv(k, v, klens)
    def forward():
        if fixed:
            batch = len(qlens)
            out, lse, *_ = flash_attn_func(
                q.view(batch, qlens[0], heads, 192),
                k.view(batch, klens[0], heads, 192),
                v.view(batch, klens[0], heads, 128),
                causal=causal, num_splits=1, pack_gqa=False, return_attn_probs=True)
            return out.flatten(0, 1), lse.permute(1, 0, 2).reshape(heads, -1)
        if paged:
            return flash_attn_with_kvcache(q, kc, vc, page_table=table,
                cache_seqlens=lengths, cu_seqlens_q=cq, max_seqlen_q=max(qlens),
                max_seqlen_k=max(klens), causal=causal, num_splits=1,
                pack_gqa=False, return_softmax_lse=True)
        return flash_attn_varlen_func(q, k, v, cq, ck, max(qlens), max(klens),
                                     causal=causal, num_splits=1, return_attn_probs=True)
    def check(out, lse):
        qo = ko = 0
        for qn, kn in zip(qlens, klens):
            if qn:
                rows = torch.arange(qn, device="cuda") if qn <= 128 else torch.tensor(
                    [0, qn // 2, qn - 1], device="cuda")
                qr, kr, vr = q[qo + rows].float(), k[ko:ko + kn].float(), v[ko:ko + kn].float()
                valid = torch.arange(kn, device="cuda")[None, :] <= rows[:, None] + kn - qn
                if not causal:
                    valid.fill_(True)
                scores = torch.einsum("rhd,khd->rhk", qr * (192 ** -.5), kr)
                scores.masked_fill_(~valid[:, None, :], float("-inf"))
                probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
                expected = torch.einsum("rhk,khd->rhd", probabilities, vr)
                torch.testing.assert_close(out[qo + rows].float(), expected, atol=.015, rtol=.015)
                torch.testing.assert_close(lse[:, qo + rows].T, torch.logsumexp(scores, dim=-1),
                                           atol=.0001, rtol=.0001)
            qo += qn
            ko += kn
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        out, lse, *_ = forward()
        check(out, lse)
        if len(qlens) > 1 or fixed:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, lse, *_ = forward()
            # Change both data and per-sequence offsets without changing the
            # packed allocation or the host maximum sequence lengths.
            q.mul_(.75)
            k.neg_()
            v.mul_(.5)
            qlens, klens = qlens[::-1], klens[::-1]
            cq.copy_(cumulative(qlens))
            ck.copy_(cumulative(klens))
            if paged:
                refresh(klens, reverse_pages=True)
            graph.replay()
            check(out, lse)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("seqlen_q,seqlen_k,heads", [
    (2048, 2048, 8), (2048, 32768, 8), (1024, 90112, 24),
], ids=["guarded", "peeled", "peeled-kv-partition"])
@pytest.mark.parametrize("layout", ["fixed", "packed", "paged64"])
@pytest.mark.parametrize("batch", [1, 2])
def test_mha_wi_bf16_large_values(seqlen_q, seqlen_k, heads, layout, batch):
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("WI requires the qualified device")
    if DISABLE_HDIM192:
        pytest.skip("D192 was disabled at build time")
    if layout == "paged64" and DISABLE_PAGEDKV:
        pytest.skip("Paged KV was disabled at build time")
    # Keep the stable FP32 accumulation finite even when batching changes the
    # selected KV partition count. Deferred probabilities still overflow PV.
    large = 2.**112 if batch == 1 else 2.**108
    qlens = [seqlen_q] * batch
    klens = [seqlen_k] * batch
    if batch > 1 and layout != "fixed":
        qlens[-1] += -1 if seqlen_q == seqlen_k else 1
        klens[-1] -= 1
    qprefix, kprefix = [0], [0]
    for qn, kn in zip(qlens, klens):
        qprefix.append(qprefix[-1] + qn)
        kprefix.append(kprefix[-1] + kn)
    q = torch.zeros(sum(qlens), heads, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(sum(klens), heads, 192, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(sum(klens), heads, 128, device="cuda", dtype=torch.bfloat16)
    q[:, :, 0] = 1
    k[:, :, 0] = 18
    for offset in kprefix[:-1]:
        k[offset:offset + 128, :, 0] = 6
    v[:, -1, -1] = large
    cq = torch.tensor(qprefix, device="cuda", dtype=torch.int32)
    ck = torch.tensor(kprefix, device="cuda", dtype=torch.int32)
    kwargs = dict(causal=True, softmax_scale=1., num_splits=1, pack_gqa=False)
    values = v
    if layout == "fixed":
        def run():
            return flash_attn_func(q.view(batch, seqlen_q, heads, 192),
                k.view(batch, seqlen_k, heads, 192), v.view(batch, seqlen_k, heads, 128),
                return_attn_probs=True, **kwargs)
    elif layout == "packed":
        def run():
            return flash_attn_varlen_func(q, k, v, cq, ck, max(qlens), max(klens),
                                          return_attn_probs=True, **kwargs)
    else:
        kc, vc, table, lengths, _ = _mha_wi_paged_kv(k, v, klens)
        values = vc
        def run():
            return flash_attn_with_kvcache(q, kc, vc, page_table=table,
                cache_seqlens=lengths, cu_seqlens_q=cq, max_seqlen_q=max(qlens),
                return_softmax_lse=True, **kwargs)
    def check(result, value, head):
        out, lse = result[:2]
        assert torch.isfinite(out).all()
        expected = torch.zeros_like(out, dtype=torch.float32)
        expected[..., head, -1] = 1
        torch.testing.assert_close(out.float() / value, expected, atol=.015, rtol=.015)
        for bi, (qn, kn) in enumerate(zip(qlens, klens)):
            count = torch.arange(qn, device="cuda").float() + kn - qn + 1
            first = count.clamp(max=128)
            shifted = torch.logaddexp(first.log() + 6, (count - first).log() + 18)
            expected_lse = torch.where(q[qprefix[bi]:qprefix[bi + 1], :, 0].T.bool(),
                                       shifted[None, :], count.log()[None, :])
            actual_lse = lse[bi] if layout == "fixed" else lse[:, qprefix[bi]:qprefix[bi + 1]]
            torch.testing.assert_close(actual_lse, expected_lse, atol=.0001, rtol=.0001)
    check(run(), large, heads - 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    # Reuse one graph while changing batch/tile/head location, value sign and
    # retry density. Dense/sparse/empty worklists must never reuse stale slots.
    for value, pattern, head in [(large, "all", heads - 1), (1., "all", heads - 1),
            (-large, "last", 0), (.25, "none", 0), (large, "first", 0),
            (large, "all", heads - 1), (1., "none", heads - 1)]:
        values.zero_()
        values[..., head, -1] = value
        q.zero_()
        if pattern == "all":
            q[:, :, 0] = 1
        elif pattern == "last":
            start = qprefix[-2] + ((qlens[-1] - 1) // 256) * 256
            q[start:qprefix[-1], head, 0] = 1
        elif pattern == "first":
            q[:min(256, qlens[0]), head, 0] = 1
        graph.replay()
        check(result, value, head)


@pytest.mark.parametrize("seqlen_q,seqlen_k,heads,splits", [
    (32, 8192, 6, 8), (1024, 90112, 24, 2),
], ids=["ws-scalar-merge", "wi-vector-merge"])
@pytest.mark.parametrize("layout", ["fixed", "packed", "paged64"])
def test_mha_wi_bf16_split_large_values(seqlen_q, seqlen_k, heads, splits, layout):
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("WI/WS requires the qualified device")
    if DISABLE_HDIM192 or (layout == "paged64" and DISABLE_PAGEDKV):
        pytest.skip("Required D192/paged support was disabled at build time")
    q = torch.zeros(seqlen_q, heads, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(seqlen_k, heads, 192, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(seqlen_k, heads, 128, device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    # One unit probability per partition keeps each PV accumulator finite.
    # Equal partition LSEs expose overflow only in the subsequent split merge.
    step = seqlen_k // splits
    k[::step, :, 0] = 256
    cq = torch.tensor([0, seqlen_q], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, seqlen_k], device="cuda", dtype=torch.int32)
    kwargs = dict(causal=True, softmax_scale=1., num_splits=1, pack_gqa=False)
    if layout == "paged64":
        kc, vc, table, lengths, refresh = _mha_wi_paged_kv(k, v, [seqlen_k])
        # Keep the capacity exact so the internal KV split boundaries match
        # the peaked keys above; workspace-capacity behavior is tested below.
        table = table[:, :seqlen_k // 64]
    def run():
        if layout == "fixed":
            return flash_attn_func(q[None], k[None], v[None], return_attn_probs=True, **kwargs)
        if layout == "packed":
            return flash_attn_varlen_func(q, k, v, cq, ck, seqlen_q, seqlen_k,
                                          return_attn_probs=True, **kwargs)
        return flash_attn_with_kvcache(q, kc, vc, page_table=table,
            cache_seqlens=lengths, cu_seqlens_q=cq, max_seqlen_q=seqlen_q,
            return_softmax_lse=True, **kwargs)
    large = 2.**127
    def set_values(signs):
        v.zero_()
        for split, sign in enumerate(signs):
            v[split * step, :, -1] = sign * large
        if layout == "paged64":
            refresh([seqlen_k])
    def check(result, signs):
        out, lse = result[:2]
        assert torch.isfinite(out).all()
        expected = torch.zeros_like(out, dtype=torch.float32)
        expected[..., -1] = sum(signs) / splits
        torch.testing.assert_close(out.float() / large, expected, atol=.015, rtol=.015)
        torch.testing.assert_close(lse, torch.full_like(lse, 256 + math.log(splits)),
                                   atol=.0001, rtol=.0001)
    signs = [1] * splits
    set_values(signs)
    check(run(), signs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for signs in [[1, 1, -1] + [0] * (splits - 3) if splits > 2 else [1, -1],
                  [0] * splits, [-1] * splits]:
        set_values(signs)
        graph.replay()
        check(result, signs)


@pytest.mark.parametrize("paged", [False, True], ids=["packed", "paged64"])
def test_mha_wi_fp8_workspace(paged):
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("FP8 WI requires the qualified device")
    if DISABLE_FP8 or DISABLE_HDIM192:
        pytest.skip("D192 FP8 was disabled at build time")
    if paged and DISABLE_PAGEDKV:
        pytest.skip("Paged KV was disabled at build time")
    from flash_attn_interface import _flash_attn_forward
    batch, qn, kn, capacity = 2, 8192, 16384, 1048576
    head_dim, value_dim, heads, heads_k = 192, 128, 6, 6
    torch.manual_seed(42)
    q, k, v = [torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
               for n, h, d in [(batch * qn, heads, head_dim),
                               (batch * kn, heads_k, head_dim), (batch * kn, heads_k, value_dim)]]
    cq = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * qn
    ck = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * kn
    used_q = torch.full((batch,), qn, device="cuda", dtype=torch.int32)
    used_k = torch.full((batch,), kn, device="cuda", dtype=torch.int32)
    table = None
    if paged:
        k = k.view(-1, 64, heads_k, head_dim)
        v = v.view(-1, 64, heads_k, value_dim)
        table = torch.zeros(batch, capacity // 64, device="cuda", dtype=torch.int32)
        table[:, :kn // 64] = torch.arange(batch * kn // 64, device="cuda",
                                          dtype=torch.int32).view(batch, -1)
    def run(generic=False):
        return _flash_attn_forward(q, k, v, cu_seqlens_q=cq,
            cu_seqlens_k=None if paged else ck, seqused_q=used_q if generic else None,
            seqused_k=used_k if paged else None, page_table=table,
            max_seqlen_q=qn, max_seqlen_k=capacity, causal=True, num_splits=1, pack_gqa=False)
    reference = run(generic=True)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = run()
    torch.cuda.synchronize()
    # The optional workspace must not scale with maxK/table capacity. Include
    # output, LSE and a small scheduler allowance in the measured live peak.
    output_bytes = result[0].numel() * result[0].element_size() + result[1].numel() * 4
    assert torch.cuda.max_memory_allocated() - allocated <= output_bytes + 2 * v.numel() + 1024**2
    def check(actual, expected):
        for a, b in zip(actual[:2], expected[:2]):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, atol=.001, rtol=.001)
    check(result, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    v.copy_(-v.float())
    graph.replay()
    check(captured, run(generic=True))


@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("qlens,klens", [([8192],[16384]),
                                       ([4097,0,257],[8193,65,16385])])
@pytest.mark.parametrize("dtype,pack_gqa", [
    (torch.bfloat16, None), (torch.bfloat16, True),
    (torch.float8_e4m3fn, False), (torch.float8_e4m3fn, True),
], ids=["bf16-auto-gqa", "bf16-packed-gqa", "fp8-unpacked-gqa", "fp8-packed-gqa"])
def test_mha_paged(heads, qlens, klens, pack_gqa, dtype):
    if DISABLE_PAGEDKV:
        pytest.skip("Paged KV was disabled at build time")
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("PPU0015 specialization")
    if DISABLE_HDIM256:
        pytest.skip("D256 was disabled at build time")
    if dtype == torch.float8_e4m3fn and DISABLE_FP8:
        pytest.skip("FP8 was disabled at build time")
    from flash_attn_interface import _flash_attn_forward, flash_attn_with_kvcache
    torch.manual_seed(sum(qlens) + sum(klens) + heads)
    counts = [(n + 63) // 64 for n in klens]
    pages = sum(counts) + 7
    permutation = torch.randperm(pages, device="cuda", dtype=torch.int32)
    table = torch.zeros((len(qlens),max(counts) + 1),device="cuda",dtype=torch.int32)
    offset = 0
    for i, count in enumerate(counts):
        table[i,:count] = permutation[offset:offset+count]
        offset += count
    q = torch.randn(sum(qlens),heads,256,device="cuda",dtype=torch.bfloat16)
    k = torch.randn(pages,64,1,256,device="cuda",dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q, k, v = (x.to(dtype) for x in (q, k, v))
    scales = [torch.empty(len(qlens), 1, device="cuda").uniform_(.7, 1.3)
              for _ in range(3)] if dtype == torch.float8_e4m3fn else [None] * 3
    if dtype == torch.float8_e4m3fn and heads == 4:
        scales[0] = None
    prefix = [0]
    for n in qlens:
        prefix.append(prefix[-1] + n)
    cq = torch.tensor(prefix,device="cuda",dtype=torch.int32)
    lengths = torch.tensor(klens,device="cuda",dtype=torch.int32)
    def run(reference=False):
        if reference:
            # Explicit query lengths still select the optimized paged mainloop.
            # Gather logical KV and use the unchanged nonpaged N64 schedule,
            # where the paired-fragment optimization is disabled.
            native_k, native_v = [
                torch.cat([cache[table[i, :counts[i]].long()].flatten(0, 1)[:kn]
                           for i, kn in enumerate(klens)], dim=0)
                for cache in (k, v)
            ]
            ck = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
            return _flash_attn_forward(q, native_k, native_v,
                cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=max(qlens),
                max_seqlen_k=max(klens), causal=True, softmax_scale=.0625,
                num_splits=1, pack_gqa=pack_gqa, q_descale=scales[0],
                k_descale=scales[1], v_descale=scales[2])
        return flash_attn_with_kvcache(q,k,v,page_table=table,
            cache_seqlens=lengths,cu_seqlens_q=cq,max_seqlen_q=max(qlens),
            max_seqlen_k=max(klens),causal=True,softmax_scale=.0625,
            num_splits=1,pack_gqa=pack_gqa,q_descale=scales[0],
            k_descale=scales[1],v_descale=scales[2],return_softmax_lse=True)
    def check(result):
        out,lse = result[:2]
        baseline = run(reference=True) if dtype == torch.float8_e4m3fn else None
        for i,(qn,kn) in enumerate(zip(qlens,klens)):
            if qn == 0:
                continue
            rows = torch.tensor([0,qn//2,qn-1],device="cuda")
            ids = table[i,:counts[i]].long()
            kd = k[ids].reshape(-1,256)[:kn].float()
            vd = v[ids].reshape(-1,256)[:kn].float()
            qd = q[prefix[i]+rows].float()
            if scales[0] is not None:
                qd *= scales[0][i, 0]
            if scales[1] is not None:
                kd *= scales[1][i, 0]
                vd *= scales[2][i, 0]
            scores = torch.einsum("rhd,kd->hrk",qd,kd)*.0625
            scores.masked_fill_(torch.arange(kn,device="cuda")[None,None,:]
                               > rows[None,:,None]+kn-qn,-torch.inf)
            expected = torch.matmul(scores.softmax(-1),vd).transpose(0,1)
            actual = (out[prefix[i]+rows].float(), lse[:,prefix[i]+rows])
            reference = (expected, scores.logsumexp(-1))
            if baseline is None:
                torch.testing.assert_close(actual[0],reference[0],atol=.015,rtol=.015)
                torch.testing.assert_close(actual[1],reference[1],atol=.0001,rtol=.0001)
            else:
                generic = (baseline[0][prefix[i]+rows].float(), baseline[1][:,prefix[i]+rows])
                for value, old, ref in zip(actual, generic, reference):
                    assert torch.isfinite(value).all()
                    error, old_error = (value-ref).abs(), (old-ref).abs()
                    slack = 2 * torch.finfo(torch.float32).eps * max(1., ref.abs().max().item())
                    assert error.max().item() <= old_error.max().item() + slack
                    assert error.double().mean().item() <= old_error.double().mean().item() + slack
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        original = run()
        check(original)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = run()
            for tensor in (q, k, v):
                if dtype == torch.bfloat16:
                    tensor.normal_()
                else:
                    tensor.copy_(torch.randn(tensor.shape,device="cuda",dtype=torch.bfloat16).to(dtype))
            for scale in scales:
                if scale is not None:
                    scale.uniform_(.7, 1.3)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        assert not torch.equal(original[0],replayed[0])
        check(replayed)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32



def test_varlen_fp8_empty_kv():
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("PPU0015 specialization")
    if DISABLE_FP8 or DISABLE_HDIM192:
        pytest.skip("Required FP8/D192 support was disabled at build time")
    torch.manual_seed(31415)
    qn, kn, heads = 257, 1025, 6
    dtype = torch.float8_e4m3fn
    q = torch.randn(2 * qn, heads, 192, device="cuda").to(dtype)
    k = torch.randn(kn, heads, 192, device="cuda").to(dtype)
    v = torch.randn(kn, heads, 128, device="cuda").to(dtype)
    cq = torch.tensor([0, qn, 2 * qn], device="cuda", dtype=torch.int32)
    ck = torch.tensor([0, kn, kn], device="cuda", dtype=torch.int32)
    scales = [torch.ones(2, heads, device="cuda") for _ in range(3)]
    def run():
        return flash_attn_varlen_func(q, k, v, cq, ck, qn, kn, causal=True,
            num_splits=1, pack_gqa=False, q_descale=scales[0], k_descale=scales[1],
            v_descale=scales[2], return_attn_probs=True)
    def check_empty(result):
        # Empty rows in a nonempty varlen launch follow Softmax::finalize and
        # the generic epilogue. A globally empty K tensor is handled by the API.
        assert torch.equal(result[0][qn:], torch.zeros_like(result[0][qn:]))
        assert torch.isneginf(result[1][:, qn:]).all()
        assert torch.isfinite(result[0][:qn]).all()
        assert torch.isfinite(result[1][:, :qn]).all()
    check_empty(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        ck.copy_(torch.tensor([0, kn // 2, kn], device="cuda", dtype=torch.int32))
        graph.replay()
        direct = run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(captured[0], direct[0], atol=0, rtol=0)
    torch.testing.assert_close(captured[1], direct[1], atol=0, rtol=0)
    assert torch.isfinite(captured[1]).all()
    with torch.cuda.stream(stream):
        ck.copy_(torch.tensor([0, kn, kn], device="cuda", dtype=torch.int32))
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    check_empty(captured)

@pytest.mark.parametrize("descale_layout", ["per-head", "broadcast", "batch-broadcast", "kv-broadcast", "none"])
@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("qlens,klens", [
    ([8192], [8192]),
    ([8193], [16385]),
    ([8192, 8192], [8192, 16384]),
    ([8192, 8192], [16384, 16384]),
    ([16384, 257, 0], [32769, 4097, 0]),
    ([32768, 0, 33, 1], [32768, 0, 0, 129]),
    ([1024] * 8, [2048] * 8),
    ([8192] + [257] * 15, [8192] + [513] * 15),
], ids=["batch1", "batch1-tail", "batch2-prefix", "batch2-fixed", "batch3-ragged",
        "batch4-empty", "batch8", "batch16-ragged"])
@pytest.mark.parametrize("layout", ["fixed", "packed", "paged64"])
def test_mha_wi_fp8(heads, qlens, klens, descale_layout, layout):
    if layout == "fixed" and (len(set(qlens)) != 1 or len(set(klens)) != 1):
        pytest.skip("Fixed tensors require uniform sequence lengths")
    paged = layout == "paged64"
    if paged and DISABLE_PAGEDKV:
        pytest.skip("Paged KV was disabled at build time")
    if cuda_cap != 89 or not torch.cuda.get_device_name().startswith("ZW-M890P"):
        pytest.skip("PPU0015 specialization")
    if DISABLE_FP8 or DISABLE_HDIM192:
        pytest.skip("Required FP8/D192 support was disabled at build time")
    torch.manual_seed(sum(qlens) + sum(klens) + heads)
    dtype = torch.float8_e4m3fn
    q, k, v = [torch.randn(n, heads, d, device="cuda", dtype=torch.bfloat16).to(dtype)
               for n, d in [(sum(qlens), 192), (sum(klens), 192), (sum(klens), 128)]]
    batch = len(qlens)
    scale_rows = 1 if descale_layout in ("broadcast", "batch-broadcast") else batch
    scale_cols = 1 if descale_layout in ("broadcast", "kv-broadcast") else heads
    scale_storage = [torch.empty(scale_rows, scale_cols, device="cuda").uniform_(.7, 1.3)
                     for _ in range(3)] if descale_layout != "none" else []
    scales = [s.expand(batch, heads) for s in scale_storage] if scale_storage else [None] * 3
    if descale_layout == "kv-broadcast":
        scales[0] = None
    def cumulative(lengths):
        return torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
    cq, ck = cumulative(qlens), cumulative(klens)
    from flash_attn_interface import _flash_attn_forward
    if layout == "fixed":
        # Leave gaps between batches to exercise batch strides independently
        # of row strides in WI and V packing, including graph replay.
        fixed_inputs = [torch.empty(batch, 2 * n, heads, d, device="cuda", dtype=dtype)[:, :n]
                        for n, d in [(qlens[0], 192), (klens[0], 192), (klens[0], 128)]]
        for dst, src in zip(fixed_inputs, (q, k, v)):
            dst.copy_(src.view(dst.shape))
    if paged:
        from flash_attn_interface import flash_attn_with_kvcache
        kc, vc, table, lengths, refresh = _mha_wi_paged_kv(k, v, klens)
    def run(generic=False):
        if layout == "fixed":
            kwargs = dict(causal=True, num_splits=1, pack_gqa=False,
                          q_descale=scales[0], k_descale=scales[1], v_descale=scales[2])
            if generic:
                used_q = torch.tensor(qlens, device="cuda", dtype=torch.int32)
                return _flash_attn_forward(*fixed_inputs, seqused_q=used_q, **kwargs)
            return flash_attn_func(*fixed_inputs, return_attn_probs=True, **kwargs)
        if paged:
            if generic:
                # Explicit used-Q lengths force the original generic paged
                # kernel with the same N128 tile, pages, lengths and descales.
                return _flash_attn_forward(q, kc, vc, cu_seqlens_q=cq,
                    seqused_q=torch.tensor(qlens, device="cuda", dtype=torch.int32),
                    seqused_k=lengths, page_table=table, max_seqlen_q=max(qlens),
                    max_seqlen_k=max(klens), causal=True, num_splits=1, pack_gqa=False,
                    q_descale=scales[0], k_descale=scales[1], v_descale=scales[2])
            return flash_attn_with_kvcache(q, kc, vc, page_table=table,
                cache_seqlens=lengths, cu_seqlens_q=cq, max_seqlen_q=max(qlens),
                max_seqlen_k=max(klens), causal=True, num_splits=1, pack_gqa=False,
                q_descale=scales[0], k_descale=scales[1], v_descale=scales[2],
                return_softmax_lse=True)
        # Explicit used-Q lengths force the generic path while preserving the
        # same batch, sequence lengths and N128 tile. An extra empty sequence
        # cannot force fallback once WI supports multiple sequences.
        used_q = torch.tensor(qlens, device="cuda", dtype=torch.int32) if generic else None
        return flash_attn_varlen_func(q, k, v, cq, ck, max(qlens), max(klens),
            seqused_q=used_q, causal=True, num_splits=1, pack_gqa=False,
            q_descale=scales[0], k_descale=scales[1], v_descale=scales[2], return_attn_probs=True)
    def check(result):
        generic = run(generic=True)
        assert torch.isfinite(result[0]).all() and not torch.isnan(result[1]).any()
        qo = ko = 0
        for batch_idx, (qn, kn) in enumerate(zip(qlens, klens)):
            if qn:
                rows = torch.arange(qn, device="cuda") if qn <= 64 else torch.unique(torch.cat([
                    torch.arange(8, device="cuda"), torch.arange(qn - 8, qn, device="cuda"),
                    torch.linspace(0, qn - 1, 24, device="cuda").long()]))
                qd = q[qo + rows].float().transpose(0, 1)
                kd, vd = k[ko:ko + kn].float().transpose(0, 1), v[ko:ko + kn].float().transpose(0, 1)
                for tensor, scale in zip((qd, kd, vd), scales):
                    if scale is not None:
                        tensor.mul_(scale[batch_idx, :, None, None])
                scores = torch.matmul(qd, kd.transpose(-1, -2)) * (192 ** -.5)
                scores.masked_fill_(torch.arange(kn, device="cuda")[None, None, :]
                                   > rows[None, :, None] + kn - qn, -torch.inf)
                expected = (torch.matmul(torch.nan_to_num(scores.softmax(-1), nan=0.), vd).transpose(0, 1),
                            scores.logsumexp(-1))
                if layout == "fixed":
                    selected = ((result[0][batch_idx, rows].float(), generic[0][batch_idx, rows].float()),
                                (result[1][batch_idx, :, rows], generic[1][batch_idx, :, rows]))
                else:
                    selected = ((result[0][qo + rows].float(), generic[0][qo + rows].float()),
                                (result[1][:, qo + rows], generic[1][:, qo + rows]))
                # Preserve the existing generic-FP8 max/mean error gate per
                # sequence for both O and LSE. Check masked rows separately so
                # infinities cannot inflate the finite-value tolerance.
                for ref, (actual, baseline) in zip(expected, selected):
                    finite = torch.isfinite(ref)
                    assert torch.equal(torch.isneginf(actual), torch.isneginf(ref))
                    assert torch.equal(torch.isneginf(baseline), torch.isneginf(ref))
                    if finite.any():
                        err, old_err = (actual[finite] - ref[finite]).abs(), (baseline[finite] - ref[finite]).abs()
                        slack = 2 * torch.finfo(torch.float32).eps * max(1., ref[finite].abs().max().item())
                        assert err.max().item() <= old_err.max().item() + slack
                        assert err.double().mean().item() <= old_err.double().mean().item() + slack
            qo += qn
            ko += kn
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        check(run())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            for x in (q, k, v):
                x.copy_(torch.randn(x.shape, device="cuda", dtype=torch.bfloat16).to(dtype))
            for scale in scale_storage:
                scale.uniform_(.7, 1.3)
            if layout == "fixed":
                for dst, src in zip(fixed_inputs, (q, k, v)):
                    dst.copy_(src.view(dst.shape))
            # Move empty/ragged sequences between batch slots without changing
            # buffer sizes or the host maximum sequence lengths.
            qlens, klens = qlens[::-1], klens[::-1]
            cq.copy_(cumulative(qlens))
            ck.copy_(cumulative(klens))
            if paged:
                refresh(klens, reverse_pages=True)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        check(captured)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

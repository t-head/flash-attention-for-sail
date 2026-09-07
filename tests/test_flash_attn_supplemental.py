"""FlashAttention supplemental UTs.

Each UT occupies one row in CASES; add or remove coverage by modifying only this list.
Parameters are passed directly to tests/test_flash_attn.py::test_flash_attn_output,
which validates forward output and backward dQ/dK/dV using the main test's numerical tolerances.

Run from the repository root:
    python -m pytest tests/test_flash_attn_supplemental.py -v
Append -k <case id> to select an individual UT; ACTest runs the entire file.
"""

from typing import NamedTuple

import pytest
import torch


class Case(NamedTuple):
    id: str
    params: dict


# head_dim=203 is not divisible by 8 and exercises the boundary path for uneven widths.
# This is supplemental parameter coverage; the original issue was not a reproducible kernel defect.
CASES = [
    # id                              parameters
    Case("hdim203_bf16_causal_fwd_bwd", dict(seqlen_q=4, seqlen_k=4, d=203, dropout_p=0.0, softcap=0.0, causal=True, local=False, alibi=False, deterministic=False, mha_type="mha", dtype=torch.bfloat16, kvpacked=False)),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_supplemental(case):
    # Import locally to avoid collecting the original parametrized test twice.
    from test_flash_attn import test_flash_attn_output as check_output

    # The reused helper seeds all visible CUDA generators.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        check_output(**case.params)

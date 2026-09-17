"""Mask constructors for the test suite and diagnostic tools.

These are FIXTURES, not part of the module's public surface: callers of
flash_attn_flex_flash hand in whatever boolean mask their model produces, and
turning it into slices is the module's own business.  Keeping the constructors
here makes that boundary explicit and keeps them out of the runtime import
path.
"""

import torch


def make_causal_mask(seqlen_q: int, seqlen_k: int, device=None) -> torch.Tensor:
    """Create a standard causal mask (lower triangle): attend where k <= q."""
    q = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k = torch.arange(seqlen_k, device=device).unsqueeze(0)
    return k <= q


def make_invcausal_mask(seqlen_q: int, seqlen_k: int, device=None) -> torch.Tensor:
    """Create an inverse-causal mask (upper triangle): attend where k >= q."""
    q = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k = torch.arange(seqlen_k, device=device).unsqueeze(0)
    return k >= q


def make_sliding_window_mask(seqlen_q: int, seqlen_k: int,
                             window_left: int, window_right: int,
                             device=None) -> torch.Tensor:
    """Create a sliding window (bicausal) mask."""
    q = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k = torch.arange(seqlen_k, device=device).unsqueeze(0)
    return (k >= q - window_left) & (k <= q + window_right)


def make_stair_mask(seqlen_q: int, seqlen_k: int, step: int,
                    device=None) -> torch.Tensor:
    """Create a stair-step mask.

    Each 'step' rows form a block where all rows attend to the same K range.
    Row group g (rows [g*step, (g+1)*step)) attends to K columns [0, (g+1)*step).
    """
    q = torch.arange(seqlen_q, device=device).unsqueeze(1)
    k = torch.arange(seqlen_k, device=device).unsqueeze(0)
    k_end = torch.minimum((q // step + 1) * step,
                          torch.full_like(q, seqlen_k))
    return k < k_end

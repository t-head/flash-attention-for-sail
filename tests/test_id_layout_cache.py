"""Cache-semantics test for the identity-level layout cache in
interface._mask_layout (fwd/bwd reuse of the decomposition).

Validates the hit/miss rules without the CUDA kernel: _mask_layout only
needs decompose_mask_optimized + build_merge_layout, both of which run on
a bare CUDA context.  The corner cases that MUST stay correct:
  - in-place mask mutation  -> miss (re-decompose), never stale layout
  - id reuse after the cached mask died -> miss (weakref identity check)
  - distinct objects / shapes -> miss
  - LRU eviction keeps the cache bounded
"""
import os
import sys
import time

import torch

_HOPPER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOPPER not in sys.path:
    sys.path.insert(0, _HOPPER)
elif sys.path.index(_HOPPER) != 0:
    sys.path.remove(_HOPPER)
    sys.path.insert(0, _HOPPER)

from flex_flash_attention import interface as itf    # noqa: E402

DEV = torch.device('cuda')
passed = 0
failed = 0


def chk(name, cond):
    global passed, failed
    print(("  PASS: " if cond else "  FAIL: ") + name)
    if cond:
        passed += 1
    else:
        failed += 1


def stair(S, P, C):
    row = torch.arange(S, device=DEV)
    col = torch.arange(S, device=DEV)
    lo = ((row // P - C) * P).clamp(min=0)
    return (col.view(1, -1) >= lo.view(-1, 1)) & \
           (col.view(1, -1) <= row.view(-1, 1))


print("=" * 74)
print("_mask_layout identity cache semantics")
print("=" * 74)

itf.flash_attn_flex_flash_cache_clear()

# 1. Same object: second call hits the cache (same dict returned).
m = stair(1024, 128, 2)
t1, s1 = itf._mask_layout(m, 1024, 1024, DEV)
t2, s2 = itf._mask_layout(m, 1024, 1024, DEV)
chk("same object -> hit (identical tensors dict)", t2 is t1 and s2 is s1)

# 2. In-place mutation bumps _version -> must re-decompose, and the new
#    layout must actually reflect the new content.
m_orig_slices = len(s1)
m[:, :] = True                      # full mask: 8 aligned slices vs stair
t3, s3 = itf._mask_layout(m, 1024, 1024, DEV)
golden = itf._mask_layout(torch.ones(1024, 1024, dtype=torch.bool,
                                     device=DEV), 1024, 1024, DEV)[1]
chk("in-place mutation -> miss + correct new layout",
    t3 is not t1 and s3 != s1 and len(s3) == len(golden))

# 3. id reuse after the cached mask died: plant a stale entry under the
#    id a NEW tensor is about to use; the weakref identity check must
#    reject it instead of serving the dead object's layout.
dead = stair(512, 128, 1)
dead_id = id(dead)
itf._mask_layout(dead, 512, 512, DEV)   # cache under dead's id
del dead
fresh = stair(512, 64, 0)               # new object (may reuse the id)
ent = itf._id_layout_cache.get(id(fresh))
if ent is not None and ent[0]() is not fresh:
    # id was reused — the guard must fire
    t4, s4 = itf._mask_layout(fresh, 512, 512, DEV)
    gold_s = itf.decompose_mask_optimized(fresh, 128, True)
    chk("id reuse after death -> weakref guard rejects stale entry",
        s4 == gold_s)
else:
    # id not reused this time; simulate by planting a weakref to a dead
    # proxy under fresh's id directly.
    import weakref
    ghost_holder = stair(512, 128, 1)
    ref = weakref.ref(ghost_holder)
    ghost_shape = ghost_holder.shape
    ghost_t, ghost_s = itf._mask_layout(ghost_holder, 512, 512, DEV)
    del ghost_holder
    itf._id_layout_cache[id(fresh)] = (ref, 0, ghost_shape,
                                       ghost_t, ghost_s)
    t4, s4 = itf._mask_layout(fresh, 512, 512, DEV)
    gold_s = itf.decompose_mask_optimized(fresh, 128, True)
    chk("dead weakref under live id -> guard rejects stale entry",
        s4 == gold_s)
del fresh

# 4. Distinct object, same content -> miss is fine (identity cache is not
#    a content cache), but it must still return the right layout.
a = stair(1024, 128, 2)
b = stair(1024, 128, 2)
ta, sa = itf._mask_layout(a, 1024, 1024, DEV)
tb, sb = itf._mask_layout(b, 1024, 1024, DEV)
chk("distinct objects -> independent correct layouts",
    sa == sb and tb is not ta)

# 5. LRU bound: more distinct masks than the cap keeps the cache small.
itf.flash_attn_flex_flash_cache_clear()
holders = [stair(256, 64, i % 3) for i in range(8)]
for x in holders:
    itf._mask_layout(x, 256, 256, DEV)
chk(f"LRU bound ({len(itf._id_layout_cache)} <= "
    f"{itf._ID_LAYOUT_CACHE_MAX})",
    len(itf._id_layout_cache) <= itf._ID_LAYOUT_CACHE_MAX)
# the last entry must still hit
tl1, sl1 = itf._mask_layout(holders[-1], 256, 256, DEV)
tl2, sl2 = itf._mask_layout(holders[-1], 256, 256, DEV)
chk("most recent entry survives LRU and hits", tl2 is tl1)
del holders, a, b, m

# 6. Production-size fwd/bwd simulation: first call decomposes, second
#    call (same object, as bwd receives it) is a pure cache hit.
itf.flash_attn_flex_flash_cache_clear()
S, P, C = 25286, 2048, 5
big = stair(S, P, C)
torch.cuda.synchronize()
t0 = time.perf_counter()
itf._mask_layout(big, S, S, DEV)          # fwd
torch.cuda.synchronize()
t_first = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
itf._mask_layout(big, S, S, DEV)          # bwd
torch.cuda.synchronize()
t_second = (time.perf_counter() - t0) * 1000
chk(f"bwd reuse at S={S}: {t_first:.2f} ms -> {t_second:.2f} ms",
    t_second < t_first / 5)
del big

itf.flash_attn_flex_flash_cache_clear()
print(f"\nRESULT: passed={passed} failed={failed}")
sys.exit(1 if failed else 0)

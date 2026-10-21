import pytest
import torch

# Expected path is asserted against actual host dispatch, not inferred from shape.
CASES = [
    (2, 128, 128, 8, 8, 64, "single"),
    (1, 1, 16384, 1, 1, 64, "split"),
    (1, 1, 1024, 8, 2, 64, "gqa"),
    (1, 65, 97, 2, 2, 96, None),
    (1, 3, 1024, 1, 1, 104, None),
    (1, 17, 23, 2, 2, 32, None),
    (1, 17, 23, 2, 2, 256, None),
    (1, 0, 7, 1, 1, 64, "empty"),
    (1, 7, 0, 2, 2, 64, "empty"),
]


def reference(q, k, v, causal):
    q, k, v = [x.transpose(1, 2).float().cpu() for x in (q, k, v)]
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if causal:
        mask = (
            torch.arange(k.shape[-2])[None, :]
            <= torch.arange(q.shape[-2])[:, None] + k.shape[-2] - q.shape[-2]
        )
        scores.masked_fill_(~mask, float("-inf"))
    probs = torch.nan_to_num(scores.softmax(-1))
    return (probs @ v).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("causal", [False, True])
def test_workspace_graph(abi, dtype, case, causal):
    torch.manual_seed(17)
    b, sq, sk, h, hk, d, path = case
    if path == "single":
        # Enough independent work on either PPU family; still assert dispatch.
        h = hk = 2 * torch.cuda.get_device_properties(0).multi_processor_count
    # Padded rows and heads exercise non-contiguous positive strides.
    tensors = [
        torch.randn(b, s, hx * 2, d, device="cuda", dtype=dtype)[:, :, ::2, :]
        for s, hx in [(sq, h), (sk, hk), (sk, hk), (sq, h)]
    ]
    q, k, v, o = tensors
    o.fill_(float("nan"))
    output_storage = o._base
    output_storage[:, :, 1::2, :].fill_(123)
    code = int(dtype == torch.bfloat16)
    dims = (b, sq, sk, h, hk, d, d)
    size = abi.query(dims)
    assert abi.query(dims, tune=True) >= size
    arena = torch.full((size + 512,), 165, device="cuda", dtype=torch.uint8)
    ptr = arena.data_ptr() + 256
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run(capacity=size, workspace=ptr, **kwargs):
        err = abi.call(tensors, workspace, capacity, stream.cuda_stream, code, causal, **kwargs)
        if err:
            raise RuntimeError(err.decode())

    def check():
        torch.testing.assert_close(
            o.float().cpu(),
            reference(q, k, v, causal),
            atol=0.035 if code else 0.005,
            rtol=0.035 if code else 0.005,
        )
        assert bool((output_storage[:, :, 1::2, :] == 123).all())
        assert bool((arena[:256] == 165).all())
        assert bool((arena[256 + size :] == 165).all())

    # Query capacity is passed unchanged. A smaller split budget may legally
    # choose fewer splits; only below the single-split minimum must fail.
    minimum = abi.query(dims, splits=1)
    for workspace, capacity in [
        (0, size),
        (ptr, 0),
        (ptr, 8),
        (ptr + 1, size),
        (ptr, 2**64 - 1),
        (ptr, minimum - 1),
    ]:
        with pytest.raises(RuntimeError):
            run(capacity, workspace)
    abi.dispatch()  # Clear the previous capture's host dispatch observation.
    with torch.cuda.stream(stream):
        run()
        observed = abi.dispatch()
        print(
            "DISPATCH",
            dims,
            path,
            causal,
            dtype,
            "capacity",
            size,
            "observed",
            observed,
            flush=True,
        )
        if path == "empty":
            assert observed[:2] == (-1, -1)
        if path == "single":
            assert observed == (1, 0, 0)
        if path == "split":
            assert observed[0] > 1 and observed[1] == 0
        # Causal sq=1 is normalized by the library before GQA selection.
        if path == "gqa":
            assert observed[2] == 1 and observed[1] == 0 and observed[0] >= 1
        stream.synchronize()
        check()
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        stream.synchronize()
        assert (
            torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        ), "raw ABI used the Torch device allocator"
        graph = torch.cuda.CUDAGraph()
        # Begin capture on the external stream, then restore the default stream
        # for the raw call. Ignoring the ABI stream would leave an empty graph.
        graph.capture_begin()
    try:
        run()
    finally:
        with torch.cuda.stream(stream):
            graph.capture_end()
    with torch.cuda.stream(stream):
        for replay in range(3):
            # Update K and V as well as Q: comparing only unchanged inputs can
            # miss stale graph pointers. All allocations remain alive.
            q.add_(0.125)
            k.mul_(0.9)
            v.add_(0.25)
            graph.replay()
            stream.synchronize()
            check()


@pytest.mark.parametrize(
    "kind",
    [
        "negative_shape",
        "large_shape",
        "negative_stride",
        "last_stride",
        "null_q",
        "dtype",
        "heads",
        "span_overflow",
    ],
)
def test_invalid_layout(abi, kind):
    ts = [torch.ones(1, 3, 2, 64, device="cuda", dtype=torch.float16) for _ in range(4)]
    shape = [n for x in ts for n in x.shape]
    stride = [n for x in ts for n in x.stride()]
    pointers = [x.data_ptr() for x in ts]
    code = 0
    if kind == "negative_shape":
        shape[1] = -1
    if kind == "large_shape":
        shape[1] = 2**31
    if kind == "negative_stride":
        stride[1] = -1
    if kind == "last_stride":
        stride[3] = 2
    if kind == "null_q":
        pointers[0] = 0
    if kind == "dtype":
        code = 2
    if kind == "heads":
        shape[6] = 3
    if kind == "span_overflow":
        stride[1] = 2**63 - 1
    arena = torch.full((4096,), 165, device="cuda", dtype=torch.uint8)
    err = abi.call(
        ts,
        arena.data_ptr(),
        4096,
        torch.cuda.current_stream().cuda_stream,
        code,
        shapes=shape,
        strides=stride,
        pointers=pointers,
    )
    assert err is not None
    assert bool((arena == 165).all())


def test_query_errors(abi):
    for dims in [(0, 1, 1, 1, 1, 64, 64), (1, -1, 1, 1, 1, 64, 64), (1, 1, 1, 3, 2, 64, 64)]:
        with pytest.raises(RuntimeError):
            abi.query(dims)
    assert abi.query((1, 1, 1, 1, 1, 6, 6)) == 0

    for splits in [-1, 129]:
        with pytest.raises(RuntimeError):
            abi.query((1, 1, 1, 1, 1, 64, 64), splits=splits)
    with pytest.raises(RuntimeError):
        abi.query((2**31 - 1, 2**31 - 1, 1, 2**31 - 1, 1, 256, 256))

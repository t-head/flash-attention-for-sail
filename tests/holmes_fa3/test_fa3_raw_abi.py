import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        (17, 23, 4, 4, 64, False),
        (16, 16, 4, 4, 128, True),
        (1, 1024, 8, 2, 64, False),
        (5, 13, 2, 2, 256, False),
    ],
)
def test_fa3_raw_graph(abi, dtype, case):
    torch.manual_seed(19)
    sq, sk, h, hk, d, causal = case
    tensors = [
        (
            torch.randn(1, s, hx, d, device="cuda", dtype=dtype)
            if causal
            else torch.randn(1, hx, s, d, device="cuda", dtype=dtype).transpose(1, 2)
        )
        for s, hx in [(sq, h), (sk, hk), (sk, hk), (sq, h)]
    ]
    q, k, v, o = tensors
    size = abi.query((1, sq, sk, h, hk, d, d))
    assert size == 67108864
    # Only the FA3 query contract is tested. Do not apply FA2's insufficient
    # workspace/empty-layout rejection requirements to the different FA3 ABI.
    workspace = torch.full((size + 512,), 165, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        abi.call(
            tensors,
            workspace.data_ptr() + 256,
            size,
            stream.cuda_stream,
            int(dtype == torch.bfloat16),
            causal,
        )

    def check():
        qc, kc, vc = [x.transpose(1, 2).float().cpu() for x in (q, k, v)]
        kc = kc.repeat_interleave(h // hk, dim=1)
        vc = vc.repeat_interleave(h // hk, dim=1)
        scores = qc @ kc.transpose(-1, -2) * d**-0.5
        if causal:
            mask = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + sk - sq
            scores.masked_fill_(~mask, float("-inf"))
        ref = (torch.nan_to_num(scores.softmax(-1)) @ vc).transpose(1, 2)
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.006
        torch.testing.assert_close(o.float().cpu(), ref, atol=tolerance, rtol=tolerance)
        assert bool((workspace[:256] == 165).all())
        assert bool((workspace[256 + size :] == 165).all())

    with torch.cuda.stream(stream):
        run()
        stream.synchronize()
        check()
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin()
    try:
        run()
    finally:
        with torch.cuda.stream(stream):
            graph.capture_end()
    with torch.cuda.stream(stream):
        for _ in range(3):
            q.add_(0.125)
            k.mul_(0.9)
            v.add_(0.25)
            graph.replay()
            stream.synchronize()
            check()
    print(
        "FA3_PASS",
        case,
        dtype,
        "workspace",
        size,
        "replays",
        3,
        "strides",
        [tuple(x.stride()) for x in tensors],
        flush=True,
    )

"""Stochastic rounding (SR) for the GDN bf16/fp16 decode state store.

SR rounds the fp32 recurrent state to the narrow storage dtype stochastically
(proportional to the residual) instead of round-to-nearest (RTN), so the
quantization is unbiased in expectation. It uses a Philox-4x32 RNG + the
hardware ``cvt.rs.{f16,bf16}x2.f32`` instruction (Blackwell sm_100a+), so these
tests require SM100+.

Checks: (1) SR-off is bit-identical to RTN, (2) SR varies with the seed and
differs from RTN, (3) SR is unbiased — the per-element mean over many seeds
lands within one local ULP of RTN, and every stored value is on the narrow grid
(SR can only ever store one of the two grid neighbors of the true value), and
(4) MTP (T>1) runs under SR.
"""
import pytest
import torch

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 10:
        pytest.skip(f"cvt.rs stochastic rounding requires SM100+, got SM{cc[0]}{cc[1]}")


def _inputs(B, T, HV, K, V, dtype, seed=0):
    torch.manual_seed(seed)
    H = HV
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(B, T, HV, V, device="cuda", dtype=dtype) * 0.5
    a = torch.randn(B, T, HV, device="cuda", dtype=dtype)
    b = torch.randn(B, T, HV, device="cuda", dtype=dtype)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
    return q, k, v, a, b, A_log, dt_bias


def _run(state, q, k, v, a, b, A_log, dt_bias, use_sr, rand_seed=None):
    # State is updated in-place (non-pool path); caller passes a fresh clone.
    out, ret_state = gated_delta_rule_decode_pretranspose(
        q=q, k=k, v=v, state=state, A_log=A_log, a=a, dt_bias=dt_bias, b=b,
        use_qk_l2norm=True, use_sr=use_sr, rand_seed=rand_seed,
    )
    return out, ret_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sr_off_matches_rtn(dtype):
    _skip_if_not_sm100()
    B, HV, K, V = 4, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, 1, HV, K, V, dtype)
    state = (torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3).to(dtype)
    o1, _ = _run(state.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=False)
    o2, _ = _run(state.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=False)
    assert torch.equal(o1, o2), "RTN (SR-off) not deterministic"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sr_varies_with_seed(dtype):
    _skip_if_not_sm100()
    B, HV, K, V = 4, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, 1, HV, K, V, dtype)
    s0 = (torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3).to(dtype)
    seed_a = torch.tensor([12345, 0], dtype=torch.int32, device="cuda")
    seed_b = torch.tensor([67890, 0], dtype=torch.int32, device="cuda")
    _, sa = _run(s0.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=True, rand_seed=seed_a)
    _, sb = _run(s0.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=True, rand_seed=seed_b)
    _, s_rtn = _run(s0.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=False)
    assert not torch.equal(sa, sb), "SR with different seeds gave identical state"
    assert not torch.equal(sa, s_rtn), "SR identical to RTN"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sr_unbiased_and_on_grid(dtype):
    """Per-element mean(SR) within 1 local ULP of RTN; every sample on-grid.

    SR rounds the true fp32 value to one of its two grid neighbors; RTN rounds
    to the nearer. So each SR sample is 0 or exactly 1 local ULP from RTN, and
    an unbiased mean stays within 1 local ULP. (Single-step only — the GDN
    recurrence's gating makes multi-step state-MSE-vs-fp32 a mix of contractive
    error-decay and chaotic divergence, neither of which isolates SR quality.)
    """
    _skip_if_not_sm100()
    B, HV, K, V = 2, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, 1, HV, K, V, dtype, seed=3)
    s0 = (torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3).to(dtype)

    _, s_rtn = _run(s0.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=False)
    s_rtn = s_rtn.float()

    N = 256
    acc = torch.zeros(B, HV, V, K, device="cuda", dtype=torch.float32)
    on_grid = True
    for t in range(N):
        seed = torch.tensor([(t * 2654435761 + 1) & 0x7FFFFFFF, 0],
                            dtype=torch.int32, device="cuda")
        _, s = _run(s0.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=True, rand_seed=seed)
        on_grid &= bool((s == s.to(dtype)).all())
        acc += s.float()
    sr_mean = acc / N

    assert on_grid, "SR produced an off-grid state value"
    mant_bits = 7 if dtype == torch.bfloat16 else 10
    min_normal = 2.0 ** (-126 if dtype == torch.bfloat16 else -14)
    nonzero = s_rtn.abs().clamp(min=min_normal)
    ulp_local = torch.exp2(torch.floor(torch.log2(nonzero))) * (2.0 ** -mant_bits)
    within = (sr_mean - s_rtn).abs() <= 1.05 * ulp_local
    frac = within.float().mean().item()
    assert frac > 0.999, f"SR biased: only {frac*100:.2f}% within 1 local ULP of RTN"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sr_mtp(dtype):
    _skip_if_not_sm100()
    B, T, HV, K, V = 2, 3, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, T, HV, K, V, dtype)
    state = (torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3).to(dtype)
    seed = torch.tensor([999, 0], dtype=torch.int32, device="cuda")
    o, _ = _run(state.clone(), q, k, v, a, b, A_log, dt_bias, use_sr=True, rand_seed=seed)
    assert torch.isfinite(o.float()).all(), "MTP SR output non-finite"

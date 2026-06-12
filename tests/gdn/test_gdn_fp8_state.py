"""fp8 (E4M3) GDN decode state backend.

The recurrent state can be stored as fp8 with a per-row (per-K-block) fp32
scale (scale = amax/448). On load the state is dequantized (fp8 * scale); on
store the kernel computes a fresh per-row amax (warp-reduced over K=128),
derives the scale, and quantizes via round-to-nearest or stochastic rounding
(hardware ``cvt.rs.satfinite.e4m3x4.f32``). The IO tensors (q/k/v/output) stay
bf16/fp16. Requires SM100+ (Blackwell).
"""
import pytest
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 10:
        pytest.skip(f"fp8 GDN state requires SM100+, got SM{cc[0]}{cc[1]}")


def _inputs(B, HV, K, V, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, 1, HV, K, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(B, 1, HV, K, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(B, 1, HV, V, device="cuda", dtype=dtype) * 0.5
    a = torch.randn(B, 1, HV, device="cuda", dtype=dtype)
    b = torch.randn(B, 1, HV, device="cuda", dtype=dtype)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
    return q, k, v, a, b, A_log, dt_bias


def _make_fp8_pool(state_f32, pool_size, indices):
    """fp8 state pool + [pool,HV,V] fp32 scale pool from per-row amax/448."""
    B, HV, V, K = state_f32.shape
    sc = (state_f32.abs().amax(dim=3) / 448.0).clamp(min=1e-8)
    fp8 = (state_f32 / sc[..., None]).to(torch.float8_e4m3fn)
    pool = torch.zeros(pool_size, HV, V, K, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.zeros(pool_size, HV, V, device="cuda", dtype=torch.float32)
    pool[indices] = fp8
    scale[indices] = sc
    return pool, scale


@pytest.mark.parametrize("use_sr", [False, True])
def test_fp8_runs(use_sr):
    _skip_if_not_sm100()
    B, HV, K, V = 4, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, HV, K, V, torch.bfloat16)
    state_f32 = torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    indices = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int32)
    pool, scale = _make_fp8_pool(state_f32, 8, indices)
    seed = torch.tensor([12345, 0], dtype=torch.int32, device="cuda") if use_sr else None
    out = gated_delta_rule(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=pool, initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True, use_sr=use_sr, rand_seed=seed, state_scale=scale,
    )
    assert torch.isfinite(out.float()).all()
    assert pool.dtype == torch.float8_e4m3fn
    assert (scale[indices] > 0).all(), "scale pool not updated by the store"


def test_fp8_requires_scale():
    _skip_if_not_sm100()
    B, HV, K, V = 2, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, HV, K, V, torch.bfloat16)
    pool = torch.zeros(4, HV, V, K, device="cuda", dtype=torch.float8_e4m3fn)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    with pytest.raises((ValueError, AssertionError)):
        gated_delta_rule(
            A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
            initial_state_source=pool, initial_state_indices=indices,
            use_qk_l2norm_in_kernel=True,  # no state_scale → must error
        )


def test_fp8_tracks_fp32_reference():
    """fp8 decode output should track an fp32-state reference within e4m3."""
    _skip_if_not_sm100()
    B, HV, K, V = 2, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, HV, K, V, torch.bfloat16, seed=2)
    state_f32 = torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    indices = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    out_ref, _ = gated_delta_rule_decode_pretranspose(
        q=q, k=k, v=v, state=state_f32.clone(), A_log=A_log, a=a, dt_bias=dt_bias,
        b=b, use_qk_l2norm=True,
    )
    pool, scale = _make_fp8_pool(state_f32, 4, indices)
    out_fp8 = gated_delta_rule(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=pool, initial_state_indices=indices,
        use_qk_l2norm_in_kernel=True, state_scale=scale,
    )
    rel = (out_fp8.float() - out_ref.float()).abs().mean().item() / (
        out_ref.float().abs().mean().item() + 1e-9
    )
    assert rel < 0.15, f"fp8 output too far from fp32 reference: rel {rel}"


@pytest.mark.parametrize("use_sr", [False, True])
def test_fp8_dispatch(use_sr):
    """Through gated_delta_rule_decode_pretranspose (the SGLang-facing entry)."""
    _skip_if_not_sm100()
    B, HV, K, V = 4, 4, 128, 128
    q, k, v, a, b, A_log, dt_bias = _inputs(B, HV, K, V, torch.bfloat16, seed=5)
    state_f32 = torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    indices = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int32)
    pool, scale = _make_fp8_pool(state_f32, 8, indices)
    seed = torch.tensor([7, 0], dtype=torch.int32, device="cuda") if use_sr else None
    out, ret = gated_delta_rule_decode_pretranspose(
        q=q, k=k, v=v, state=None, A_log=A_log, a=a, dt_bias=dt_bias, b=b,
        use_qk_l2norm=True, initial_state=pool, initial_state_indices=indices,
        state_scale=scale, use_sr=use_sr, rand_seed=seed,
    )
    assert torch.isfinite(out.float()).all()
    assert ret.dtype == torch.float8_e4m3fn

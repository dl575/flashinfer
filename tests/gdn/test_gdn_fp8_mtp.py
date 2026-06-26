"""fp8 (E4M3) GDN MTP / target-verify path.

Exercises the verify path the 397B+NEXTN eval hits (`gated_delta_rule_mtp` in
gdn_decode_bf16_state, the live MTP path — NOT the legacy gdn_decode_mtp.py):

- FP32 ``intermediate_states_buffer`` accepted (assert relaxed to allow fp32).
- fp8 initial state dequantized on load (per-row scale pool).
- fp32-direct intermediate-snapshot write (no rounding) for fp8 state.
- the (B*HV>=128, T>=2) config drives the wide_vec dispatch, exercising the
  forwarding of use_sr/state_scale/philox to gated_delta_rule_mtp_wide_vec.

Requires SM100+ (Blackwell). disable_state_update=True mirrors verify (no final
store; only the per-draft-token intermediate snapshots are written).
"""
import pytest
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 10:
        pytest.skip(f"fp8 GDN state requires SM100+, got SM{cc[0]}{cc[1]}")


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


def _mtp_inputs(B, T, HV, K, V, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, HV, K, device="cuda", dtype=torch.bfloat16) * 0.3
    k = torch.randn(B, T, HV, K, device="cuda", dtype=torch.bfloat16) * 0.3
    v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16) * 0.3
    a = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
    return q, k, v, a, b, A_log, dt_bias


# (4,4) -> B*HV=16 (ILP4 path); (16,8) -> 128 (wide_vec path, exercises F-1).
@pytest.mark.parametrize("B,HV", [(4, 4), (16, 8)])
@pytest.mark.parametrize("use_sr", [False, True])
def test_fp8_mtp_runs(B, HV, use_sr):
    _skip_if_not_sm100()
    T, K, V = 2, 128, 128
    q, k, v, a, b, A_log, dt_bias = _mtp_inputs(B, T, HV, K, V, seed=B)
    state_f32 = torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    indices = torch.arange(B, dtype=torch.int32, device="cuda")
    pool, scale = _make_fp8_pool(state_f32, B + 2, indices)
    inter = torch.zeros(B, T, HV, V, K, device="cuda", dtype=torch.float32)  # FP32 (S-1)
    seed = torch.tensor([99, 0], dtype=torch.int32, device="cuda") if use_sr else None
    out = gated_delta_rule_mtp(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=pool, initial_state_indices=indices,
        intermediate_states_buffer=inter, disable_state_update=True,
        use_qk_l2norm_in_kernel=True, use_sr=use_sr, rand_seed=seed, state_scale=scale,
    )
    assert torch.isfinite(out.float()).all(), "MTP output not finite"
    assert inter.dtype == torch.float32
    assert torch.isfinite(inter).all(), "intermediate buffer has non-finite"
    assert inter.abs().sum().item() > 0, "FP32 intermediate snapshot not written"


@pytest.mark.parametrize("B,HV", [(4, 4), (16, 8)])
def test_fp8_mtp_tracks_bf16(B, HV):
    """fp8 verify output should track a bf16-state reference within e4m3 tol."""
    _skip_if_not_sm100()
    T, K, V = 2, 128, 128
    q, k, v, a, b, A_log, dt_bias = _mtp_inputs(B, T, HV, K, V, seed=B + 1)
    state_f32 = torch.randn(B, HV, V, K, device="cuda", dtype=torch.float32) * 0.3
    indices = torch.arange(B, dtype=torch.int32, device="cuda")
    bf16_pool = torch.zeros(B + 2, HV, V, K, device="cuda", dtype=torch.bfloat16)
    bf16_pool[indices] = state_f32.to(torch.bfloat16)
    bf16_inter = torch.zeros(B, T, HV, V, K, device="cuda", dtype=torch.bfloat16)
    out_ref = gated_delta_rule_mtp(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=bf16_pool, initial_state_indices=indices,
        intermediate_states_buffer=bf16_inter, disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
    )
    pool, scale = _make_fp8_pool(state_f32, B + 2, indices)
    inter = torch.zeros(B, T, HV, V, K, device="cuda", dtype=torch.float32)
    out_fp8 = gated_delta_rule_mtp(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b,
        initial_state_source=pool, initial_state_indices=indices,
        intermediate_states_buffer=inter, disable_state_update=True,
        use_qk_l2norm_in_kernel=True, state_scale=scale,
    )
    rel = (out_fp8.float() - out_ref.float()).abs().mean().item() / (
        out_ref.float().abs().mean().item() + 1e-9
    )
    assert rel < 0.15, f"fp8 MTP output too far from bf16 reference: rel {rel}"

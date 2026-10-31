import numpy as np
import pytest

from worldsim.simulator.factors import BASELINE, CONSTANTS, NAMES
from worldsim.simulator.handy import W, XC, XE, Y, SimConfig, derivatives, simulate


def _params(n=64, seed=0, **over):
    rng = np.random.default_rng(seed)
    p = dict(BASELINE)
    p.update(delta=rng.uniform(5e-6, 3e-5, n), kappa=rng.uniform(1, 8, n), elite_frac=rng.uniform(0, .3, n),
             event_rate=0.02, event_sev=0.3, temp=rng.uniform(0, 4, n), water=rng.uniform(.5, 2.5, n))
    p.update(over)
    return p


def test_non_negativity():
    r = simulate(_params(), SimConfig(horizon=300))
    assert (r.traj >= 0).all() and np.isfinite(r.traj).all()
    assert (r.traj[:, :, Y] <= CONSTANTS["lam"] + 1e-6).all()  # nature never exceeds capacity


def test_determinism_same_seed():
    a = simulate(_params(), SimConfig(horizon=200, seed=7))
    b = simulate(_params(), SimConfig(horizon=200, seed=7))
    assert np.array_equal(a.traj, b.traj) and np.array_equal(a.events, b.events)
    c = simulate(_params(), SimConfig(horizon=200, seed=8))
    assert not np.array_equal(a.traj, c.traj)


def test_events_hurt_nature_and_no_events_is_deterministic():
    p0 = _params(event_rate=0.0)
    r0a, r0b = simulate(p0, SimConfig(horizon=200, seed=1)), simulate(p0, SimConfig(horizon=200, seed=2))
    assert np.array_equal(r0a.traj, r0b.traj) and r0a.events.sum() == 0   # no randomness without events
    p1 = _params(event_rate=0.05, event_sev=0.5)
    r1 = simulate(p1, SimConfig(horizon=200, seed=1))
    assert r1.events.sum() > 0
    assert r1.traj[:, :, Y].mean() < r0a.traj[:, :, Y].mean()


def test_wealth_balance():
    """dw/dt = delta*xC*y - C_C - C_E: the stored w must match the integral of that flow (no events)."""
    p = dict(BASELINE); p.update(delta=1.2e-5)
    cfg = SimConfig(horizon=100, record_every=1, dt=0.25)
    r = simulate(p, cfg)
    pa = {k: np.atleast_1d(np.asarray(v, float)) for k, v in p.items()}
    s = r.traj[0]
    flow = []
    for st in s:
        d = derivatives(st[None], pa, dict(gamma=pa["gamma"], alpha_m=np.array([CONSTANTS["alpha_m"]])))
        flow.append(d[0, W])
    integral = np.trapezoid(flow, r.times)
    assert integral == pytest.approx(s[-1, W] - s[0, W], rel=1e-3, abs=1e-6)


def test_equilibrium_matches_theory():
    """Egalitarian baseline at delta_opt: y* = lam/2 and xC* = gamma (lam - y*) / delta."""
    r = simulate(BASELINE, SimConfig(horizon=1500))
    assert r.traj[0, -1, Y] == pytest.approx(50.0, rel=1e-3)
    assert r.traj[0, -1, XC] == pytest.approx(0.01 * 50 / 6.67e-6, rel=1e-3)
    assert not r.collapsed[0]


def test_temperature_and_water_effects():
    base = dict(BASELINE); base.update(delta=1e-5)
    hot = simulate({**base, "temp": 4.0}, SimConfig(horizon=300))
    cold = simulate({**base, "temp": 0.0}, SimConfig(horizon=300))
    assert hot.traj[0, :, Y].min() < cold.traj[0, :, Y].min() or hot.collapsed[0] >= cold.collapsed[0]
    dry = simulate({**base, "water": 0.3}, SimConfig(horizon=600))
    wet = simulate({**base, "water": 10.0}, SimConfig(horizon=600))
    assert dry.traj[0, :, XC].max() < wet.traj[0, :, XC].max()


def test_batch_equals_single():
    p = _params(n=4, event_rate=0.0)
    batch = simulate(p, SimConfig(horizon=100))
    single = simulate({k: np.atleast_1d(v)[2] if np.size(v) > 1 else v for k, v in p.items()}, SimConfig(horizon=100))
    assert np.allclose(batch.traj[2], single.traj[0])


def test_collapse_definition():
    p = dict(BASELINE); p.update(delta=6.67e-6 * 4)
    r = simulate(p, SimConfig(horizon=600))
    assert r.collapsed[0] and r.t_collapse[0] < 600
    pop = r.traj[0, :, XC] + r.traj[0, :, XE]
    i = int(r.t_collapse[0] // (6))
    before = pop[: i - 1]
    assert (before >= 0.1 * np.maximum.accumulate(before)).all()   # no collapse before t_collapse


def test_elite_famine_uses_kappa():
    """Eq. (6): alpha_E uses C_E/(s xE) = kappa*min(1,w/wth), so elites starve only when w/wth <= 1/kappa."""
    from worldsim.simulator.handy import _effective_rates
    p = {k: np.atleast_1d(np.asarray(v, float)) for k, v in BASELINE.items()}
    p["kappa"] = np.array([10.0])
    eff = _effective_rates(p)
    state = np.array([[100.0, 100.0, 50.0, 0.0]])
    state[0, W] = 0.5 * CONSTANTS["rho"] * (100 + 10 * 100)          # w/w_th = 0.5: commoners starve, elites (1/kappa=0.1) do not
    d = derivatives(state, p, eff)
    assert d[0, XC] / 100 == pytest.approx(p["beta"][0] - (CONSTANTS["alpha_m"] + 0.5 * (CONSTANTS["alpha_M"] - CONSTANTS["alpha_m"])))
    assert d[0, XE] / 100 == pytest.approx(p["beta"][0] - CONSTANTS["alpha_m"])


def test_paper_scenarios_qualitative():
    """Type-L: nature recovers while population dies; Type-N: nature and population both reach 0 (irreversible)."""
    D = 6.67e-6
    L = simulate({**BASELINE, "delta": D, "kappa": 100.0}, SimConfig(horizon=1500), init=dict(xc0=100.0, xe0=1e-3))
    assert L.collapsed[0] and L.traj[0, -1, XC] < 1.0 and L.traj[0, -1, Y] > 90
    N = simulate({**BASELINE, "delta": 12 * D}, SimConfig(horizon=1500))
    assert N.traj[0, -1, Y] < 1e-6 and N.traj[0, -1, XC] < 1.0
    cyc = simulate({**BASELINE, "delta": 4 * D}, SimConfig(horizon=1500))
    assert cyc.traj[0, -1, Y] > 1.0 or cyc.traj[0, :, XC].max() > 1e4   # reversible cycles at 4x with RK4


def test_euler_close_to_rk4_for_small_dt():
    p = dict(BASELINE); p.update(delta=1e-5)
    a = simulate(p, SimConfig(horizon=200, dt=0.05, record_every=100, integrator="euler"))
    b = simulate(p, SimConfig(horizon=200, dt=1.0, record_every=5, integrator="rk4"))
    assert np.allclose(a.traj[0, -1], b.traj[0, -1], rtol=2e-2)


def test_factor_ranges_match_dataset_config():
    """The factor registry (used by counterfactuals and the app) must use the same ranges as the dataset."""
    import pathlib
    import yaml
    from worldsim.simulator.factors import FACTORS
    cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "configs/dataset.yaml", encoding="utf8"))["ranges"]
    for f in FACTORS:
        assert (f.low, f.high) == pytest.approx(tuple(cfg[f.name])), f.name

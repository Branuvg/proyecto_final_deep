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

import sys; sys.path.insert(0, "src")
import numpy as np
from worldsim.simulator.handy import simulate, SimConfig
from worldsim.simulator.factors import BASELINE
D = 6.67e-6
def run(name, over, init=None, integ="euler", dtype="float32", T=1000):
    p = {**BASELINE, **over}
    r = simulate(p, SimConfig(horizon=T, record_every=5, integrator=integ, dtype=dtype), init=init)
    xc, xe, y, w = (r.traj[0, :, i].astype(float) for i in range(4))
    print(f"{name:34s} {integ:5s} {dtype}: min y {y.min():.1e} y_end {y[-1]:.1e} pop_peak {(xc+xe).max():9.0f} pop_end {(xc+xe)[-1]:9.1f} xE_end {xe[-1]:.1f} collapsed {r.collapsed[0]} t {r.t_collapse[0]:.0f}")
for m in (1, 2.5, 4, 5.5):
    run(f"egal {m}x", dict(delta=D*m))
run("egal 5.5x", dict(delta=D*5.5), integ="rk4", dtype="float64")
run("egal 5.5x", dict(delta=D*5.5), integ="euler", dtype="float64")
run("unequal TypeL k100 xE=1e-3", dict(delta=D, kappa=100), init=dict(xc0=100., xe0=1e-3), T=1500)
run("unequal TypeN k100 xE=.2 d=1e-4", dict(delta=1e-4, kappa=100), init=dict(xc0=100., xe0=0.2))
run("unequal TypeN (rk4)", dict(delta=1e-4, kappa=100), init=dict(xc0=100., xe0=0.2), integ="rk4", dtype="float64")
run("equit k1 xE=25", dict(delta=D, kappa=1), init=dict(xc0=100., xe0=25.))

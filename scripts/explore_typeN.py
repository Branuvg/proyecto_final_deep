import sys; sys.path.insert(0, "src")
import numpy as np
from worldsim.simulator.handy import simulate, SimConfig
from worldsim.simulator.factors import BASELINE
D = 6.67e-6
def irreversible(mult, integ, dtype):
    p = {**BASELINE, "delta": D*mult}
    r = simulate(p, SimConfig(horizon=2000, record_every=10, integrator=integ, dtype=dtype))
    pop = r.traj[0,:,0]; y = r.traj[0,:,2]
    return bool(y[-1] < 1e-6 and pop[-1] < 1.0)
for integ, dtype in (("rk4","float64"),("euler","float32")):
    grid = np.arange(2, 20.01, 0.25)
    flags = [irreversible(m, integ, dtype) for m in grid]
    first = next((m for m, f in zip(grid, flags) if f), None)
    print(integ, dtype, "first irreversible multiple of delta*:", first, "| monotone above:", all(f for m,f in zip(grid,flags) if first and m>=first))

import sys, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from worldsim.data.sampling import sample_params
from worldsim.simulator.handy import simulate, SimConfig
R = eval(sys.argv[1]) if len(sys.argv)>1 else {}
p = sample_params(6000, 0, R)
r = simulate(p, SimConfig(seed=0))
print("survival", round(1-r.collapsed.mean(),3), "t_col q", np.percentile(r.t_collapse[r.collapsed],[5,25,50,75,95]))
df = pd.DataFrame(p); df["surv"]=~r.collapsed
for c in p:
    q = pd.qcut(df[c],5,duplicates="drop"); print(f"{c:11s}", df.groupby(q,observed=True)["surv"].mean().round(2).tolist())

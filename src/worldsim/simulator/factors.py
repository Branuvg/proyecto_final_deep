"""Factor registry: the 11 planet factors, their sampling ranges and whether they are actionable.

*Actionable* factors are those a decision maker could plausibly change (policy levers);
*non-actionable* ones are environmental/biophysical givens. Phase 2 counterfactuals may
only move actionable factors.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Factor:
    name: str
    low: float
    high: float
    actionable: bool
    log: bool  # sample uniformly in log-space
    desc: str


# Order matters: it is the column order of every feature matrix in the project.
# Ranges here must equal configs/dataset.yaml (checked by tests/test_simulator.py::test_factor_ranges_match_dataset_config).
FACTORS: tuple[Factor, ...] = (
    Factor("delta", 5e-06, 6e-05, True, True, "Depletion factor of nature per commoner (HANDY delta)"),
    Factor("kappa", 1, 3, True, True, "Inequality: elite consumption multiplier (HANDY kappa)"),
    Factor("s", 0.00035, 0.00065, True, False, "Subsistence salary per capita (HANDY s)"),
    Factor("water_eff", 0, 0.6, True, False, "Water-use efficiency: fraction of water demand saved"),
    Factor("beta", 0.025, 0.045, False, False, "Birth rate, beta_C = beta_E (HANDY beta)"),
    Factor("gamma", 0.004, 0.02, False, False, "Nature regeneration rate (HANDY gamma)"),
    Factor("temp", 0, 5, False, False, "Temperature anomaly in deg C (extension)"),
    Factor("water", 0.05, 1.2, False, False, "Water supply relative to reference demand (extension)"),
    Factor("event_rate", 0, 0.12, False, False, "Extreme-event rate, events/year (Poisson, extension)"),
    Factor("event_sev", 0.1, 0.9, False, False, "Extreme-event severity in (0,1) (extension)"),
    Factor("elite_frac", 0, 0.1, False, False, "Initial elite share of the population"),
)
NAMES = [f.name for f in FACTORS]
ACTIONABLE = {f.name: f.actionable for f in FACTORS}

# Fixed HANDY constants (Motesharrei et al. 2014 values for the baseline scenarios)
CONSTANTS = dict(alpha_m=0.01, alpha_M=0.07, rho=5e-3, lam=100.0, pop0=100.0, p_ref=75000.0)
# Extension constants (documented in docs/DECISIONES.md)
EXT = dict(temp_tol=3.0, temp_mort=0.15)

# Canonical egalitarian HANDY point (no elites, no extension effects)
BASELINE = dict(delta=6.67e-6, kappa=1.0, s=5e-4, water_eff=0.0, beta=0.03, gamma=0.01,
                temp=0.0, water=1e6, event_rate=0.0, event_sev=0.0, elite_frac=0.0)

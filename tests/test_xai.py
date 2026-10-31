import numpy as np
import torch

from worldsim.xai.attributions import exact_shapley
from worldsim.xai.fidelity import agreement, deletion_auc, deletion_curve_model, deletion_orders, row_spearman


def test_exact_shapley_linear_model_is_coefficient_times_delta():
    rng = np.random.default_rng(0)
    coef = torch.tensor([1.5, -2.0, 0.5, 0.0])
    X = rng.normal(size=(20, 4)).astype(np.float32)
    phi = exact_shapley(lambda t: t @ coef, X, np.zeros(4, np.float32), torch.device("cpu"))
    assert np.allclose(phi, X * coef.numpy(), atol=1e-5)
    assert np.allclose(phi.sum(1), X @ coef.numpy(), atol=1e-5)          # efficiency


def test_row_spearman_and_agreement_perfect():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(50, 11))
    assert np.allclose(row_spearman(A, 3 * A + 1), 1.0)
    assert np.allclose(row_spearman(A, -A), -1.0)
    r = agreement(A, A, thr=0.01)
    assert r["spearman"] > 0.999 and r["sign_agreement"] == 1.0 and r["top1"] == 1.0 and r["top3_overlap"] == 1.0


def test_deletion_attribution_order_beats_random_for_additive_model():
    rng = np.random.default_rng(2)
    coef = rng.normal(size=11)
    Z = rng.normal(size=(300, 11)); phi = Z * coef                       # exact attributions of a linear model
    f = lambda z: z @ coef
    orders = deletion_orders(phi, rng)
    auc = {k: deletion_auc(deletion_curve_model(f, Z, o)) for k, o in orders.items()}
    assert auc["attribution"] < auc["random"] < 0.5 and auc["attribution"] < auc["global_importance"]

"""Step 5 - disparity statistics, BISG-weighted. No language model involved.

Group membership is never observed; each file carries `p_group_b`, a
BISG-style probability. Rates are probability-weighted: a file with p=0.7
counts 0.7 toward group B and 0.3 toward group A. This is the standard way to
use proxy estimates without hard-assigning anyone to a group.

  AIR              adverse impact ratio = approval rate B / approval rate A
  counterfactual   AIR if files in one finding had been decided by the
                   written policy - "how much of the gap is this rule?"
  regression       logistic odds ratio on p_group_b, controlling first for
                   the written-policy factors, then also for the practice
                   factors. If adding a factor absorbs the disparity, the
                   factor is what carries it - and its business justification
                   becomes the question for counsel.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def air(approve: np.ndarray, p_b: np.ndarray) -> tuple[float, float, float]:
    rate_b = float((p_b * approve).sum() / p_b.sum())
    rate_a = float(((1 - p_b) * approve).sum() / (1 - p_b).sum())
    return rate_b / rate_a, rate_a, rate_b


def air_ci(approve: np.ndarray, p_b: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(approve)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(air(approve[i], p_b[i])[0])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def counterfactual_air(approve: np.ndarray, p_b: np.ndarray, mask: np.ndarray, policy_ok: np.ndarray) -> float:
    a = approve.copy()
    a[mask] = policy_ok[mask]
    return air(a, p_b)[0]


def group_share(p_b: np.ndarray, mask: np.ndarray) -> float:
    return float(p_b[mask].mean()) if mask.any() else float("nan")


def odds_ratio(book: pd.DataFrame, p_b: np.ndarray, policy_ok: np.ndarray, extra: list[str]) -> dict:
    import statsmodels.api as sm

    cols = {"policy_approves": policy_ok.astype(float)}
    for f in ["fico", "dti", "ltv", *extra]:
        x = book[f].to_numpy(dtype=float)
        cols[f] = (x - x.mean()) / (x.std() or 1.0)
    cols["p_group_b"] = p_b
    X = sm.add_constant(pd.DataFrame(cols))
    y = book["approve"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    coef = fit.params["p_group_b"]
    lo, hi = fit.conf_int().loc["p_group_b"]
    return {"or": float(np.exp(coef)), "lo": float(np.exp(lo)), "hi": float(np.exp(hi)),
            "p": float(fit.pvalues["p_group_b"])}


def proxy_scan(book: pd.DataFrame, p_b: np.ndarray, feats: list[str]) -> list[tuple[str, float]]:
    out = []
    for f in feats:
        x = book[f].to_numpy(dtype=float)
        r = float(np.corrcoef(x, p_b)[0, 1]) if x.std() > 0 else 0.0
        out.append((f, r))
    return sorted(out, key=lambda t: -abs(t[1]))

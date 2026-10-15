"""A synthetic loan book with three planted shadow rules - the demo's ground truth.

The written policy (data/credit_policy.json) approves when FICO >= 660,
DTI <= 45 and LTV <= 95. Practice departs from it in three planted ways:

  R1  book-wide, undocumented: self-employed applicants above 38% DTI are
      declined even though policy allows up to 45%.               (stricter)
  R2  book-wide override: existing customers with FICO 640-659 are approved
      as underwriter exceptions.                                     (looser)
  R3  one branch only, flagged as overrides: applicants with under two years
      at their address are declined.                              (stricter)

Groups are abstract ("A", "B"). Group B is drawn as more often self-employed,
less often an existing customer, with shorter residence and a slightly lower
FICO mean - so every planted rule widens the approval gap, and some of the gap
is explained by a legitimate credit factor. The true group is kept for scoring
only; the pipeline sees a noisy BISG-style probability, `p_group_b`.

Stated adverse-action reasons are planted too: R1 declines usually cite DTI
(the applicant's DTI is within policy, so the notice is inaccurate), R3
declines usually cite credit score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REASONS = ["CREDIT_SCORE", "DTI", "LTV", "INCOME_VERIFICATION", "RESIDENCE_STABILITY"]


def generate(n: int = 20_000, seed: int = 7, n_branches: int = 15, shadow_branch: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    b = rng.random(n) < 0.35

    fico = np.clip(np.round(rng.normal(np.where(b, 690, 705), 55)), 300, 850).astype(int)
    dti = np.clip(rng.normal(34, 9, n), 5, 70).round(1)
    ltv = np.clip(rng.normal(80, 10, n), 20, 115).round(1)
    income = np.clip(rng.lognormal(np.log(90), 0.45, n), 15, 500).round(0)
    loan_amount = np.clip(income * rng.uniform(1.5, 5, n), 5, 1500).round(0)
    self_employed = rng.random(n) < np.where(b, 0.30, 0.12)
    existing = rng.random(n) < np.where(b, 0.25, 0.45)
    years_at_address = np.clip(rng.exponential(np.where(b, 4.0, 7.0)), 0, 40).round(1)
    branch = rng.integers(1, n_branches + 1, n)
    underwriter = branch * 10 + rng.integers(0, 4, n)

    policy = (fico >= 660) & (dti <= 45) & (ltv <= 95)
    approve = policy.copy()

    r2 = ~policy & (fico >= 640) & (fico < 660) & (dti <= 45) & (ltv <= 95) & existing & (rng.random(n) < 0.90)
    approve |= r2
    r1 = approve & self_employed & (dti > 38) & (rng.random(n) < 0.95)
    approve &= ~r1
    r3 = approve & (branch == shadow_branch) & (years_at_address < 2) & (rng.random(n) < 0.95)
    approve &= ~r3
    noise = rng.random(n) < 0.015
    approve ^= noise

    def pick(options, p):
        return rng.choice(options, size=n, p=p)

    first_fail = np.where(fico < 660, "CREDIT_SCORE", np.where(dti > 45, "DTI", np.where(ltv > 95, "LTV", "")))
    reason = np.where(
        r3, pick(["CREDIT_SCORE", "RESIDENCE_STABILITY"], [0.6, 0.4]),
        np.where(
            r1, pick(["DTI", "INCOME_VERIFICATION"], [0.75, 0.25]),
            np.where(first_fail != "", first_fail, pick(REASONS, [0.2] * 5)),
        ),
    )
    reason = np.where(approve, "", reason)

    gt_rule = np.select([noise, r3, r1, r2], ["noise", "R3", "R1", "R2"], default="")
    gt_bad = ~approve & ((r1 & (reason == "DTI")) | (r3 & (reason == "CREDIT_SCORE")))
    gt_off = ~approve & ((r1 & (reason == "INCOME_VERIFICATION")) | (r3 & (reason == "RESIDENCE_STABILITY")))

    p_b = np.where(b, rng.beta(5, 2, n), rng.beta(2, 5, n)).round(3)

    return pd.DataFrame({
        "app_id": [f"APP-{i:06d}" for i in range(n)],
        "branch": branch,
        "underwriter": underwriter,
        "fico": fico,
        "dti": dti,
        "ltv": ltv,
        "income": income,
        "loan_amount": loan_amount,
        "self_employed": self_employed.astype(int),
        "existing_customer": existing.astype(int),
        "years_at_address": years_at_address,
        "approve": approve.astype(int),
        "override": (r2 | r3).astype(int),
        "stated_reason": reason,
        "p_group_b": p_b,
        "_true_group": np.where(b, "B", "A"),
        "_gt_rule": gt_rule,
        "_gt_inaccurate_notice": gt_bad.astype(int),
        "_gt_offpolicy_notice": gt_off.astype(int),
    })

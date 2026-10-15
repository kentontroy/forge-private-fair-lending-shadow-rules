"""Application attributes the audit reasons over, and their legal domains.

Every formula the pipeline handles - the written policy, the induced practice
rules, anything TwIL-LM proposes - may only mention these names. Protected
characteristics are deliberately absent: they never enter rule induction, and
the group estimate lives in a separate table joined only in the disparity step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "int" | "real" | "bool"
    lo: float
    hi: float
    label: str
    aliases: tuple[str, ...] = ()


_FEATURES = [
    Feature("fico", "int", 300, 850, "FICO", ("ficoscore", "creditscore", "score")),
    Feature("dti", "real", 0, 100, "DTI", ("debttoincome", "debttoincomeratio", "dtiratio")),
    Feature("ltv", "real", 0, 150, "LTV", ("loantovalue", "loantovalueratio", "ltvratio")),
    Feature("income", "real", 0, 10_000, "income ($k)", ("annualincome",)),
    Feature("loan_amount", "real", 0, 10_000, "loan amount ($k)", ("loanamount", "amount")),
    Feature(
        "years_at_address", "real", 0, 100, "years at address",
        ("yearsataddress", "timeataddress", "residenceyears", "yearsatresidence"),
    ),
    Feature("self_employed", "bool", 0, 1, "self-employed", ("selfemployed", "isselfemployed")),
    Feature(
        "existing_customer", "bool", 0, 1, "existing customer",
        ("existingcustomer", "isexistingcustomer", "customer"),
    ),
]

FEATURES: dict[str, Feature] = {f.name: f for f in _FEATURES}
ORDER: list[str] = [f.name for f in _FEATURES]

# Normalised token (lowercase, alphanumerics only) -> canonical feature name.
ALIASES: dict[str, str] = {}
for _f in _FEATURES:
    ALIASES[_f.name.replace("_", "")] = _f.name
    for _a in _f.aliases:
        ALIASES[_a] = _f.name

APPLICATION_COLUMNS = ["app_id", "branch", "underwriter", *ORDER]
DECISION_COLUMNS = ["approve", "override", "stated_reason"]

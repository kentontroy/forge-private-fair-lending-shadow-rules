"""Steps 2-4 - induce the policy as practiced, find where it departs, explain overrides.

Step 2 fits a shallow decision tree to the decision history on permissible
application attributes only. Its leaves are boxes, so "the policy as
practiced", A(x), is a union of boxes in the same language as the written
policy P(x).

Step 3 intersects them: every tree leaf that approves, crossed with every box
where the policy declines (and vice versa), is a region of disagreement. Z3
then proves the region list is complete - no applicant profile where P and A
disagree is missing from it - and produces a concrete witness for each.

Step 4 looks at what the tree could *not* explain. Files whose decision
contradicts A(x) are residuals; a branch with an outsized residual rate has a
local rule of its own. A small exhaustive search (and, optionally, TwIL-LM's
in-context rule induction) states that rule, and the data validates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import z3
from sklearn.tree import DecisionTreeClassifier

from . import logic
from .logic import INF, Box, Interval
from .schema import FEATURES, ORDER


@dataclass
class Leaf:
    box: Box
    approve: bool
    n: int
    purity: float


@dataclass
class Region:
    box: Box
    direction: str  # "stricter" (policy approves, practice declines) | "looser"
    n: int
    follows_practice: float  # share of files decided the way practice predicts
    material: bool
    witness: dict | None = None
    z3_label: str = ""
    twil_label: str = ""

    @property
    def headline(self) -> str:
        if self.direction == "stricter":
            return "Policy approves, practice declines"
        return "Policy declines, practice approves"


# --------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------

def fit_practice(book: pd.DataFrame, max_depth: int = 7, min_leaf: int = 40, seed: int = 0):
    X = book[ORDER].to_numpy(dtype=float)
    y = book["approve"].to_numpy()
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(X, y)

    t, leaves = tree.tree_, []

    def walk(node: int, box: Box):
        if t.children_left[node] == -1:
            dist = t.value[node][0]
            cls = tree.classes_[int(np.argmax(dist))]
            leaves.append(Leaf(box, bool(cls == 1), int(t.n_node_samples[node]), float(dist.max() / dist.sum())))
            return
        f, thr = ORDER[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], box.intersect(Box({f: Interval(-INF, False, thr, True)})))
        walk(t.children_right[node], box.intersect(Box({f: Interval(thr, False, INF, False)})))

    walk(0, Box())
    return tree, leaves


def practice_predict(tree, book: pd.DataFrame) -> np.ndarray:
    return tree.predict(book[ORDER].to_numpy(dtype=float))


# --------------------------------------------------------------------------
# Step 3
# --------------------------------------------------------------------------

def disagreements(policy_node, leaves: list[Leaf], book: pd.DataFrame, min_files: int = 30):
    p_app = logic.disjoint(logic.to_boxes(policy_node))
    p_dec = logic.disjoint(logic.to_boxes(policy_node, negate=True))

    raw = {"stricter": [], "looser": []}
    for leaf in leaves:
        for pb in (p_dec if leaf.approve else p_app):
            r = leaf.box.intersect(pb)
            if not r.empty():
                raw["looser" if leaf.approve else "stricter"].append(r)

    approve = book["approve"].to_numpy()
    regions = []
    for direction, boxes in raw.items():
        for box in logic.merge(boxes):
            m = box.mask(book)
            n = int(m.sum())
            follows = float((approve[m] == 0).mean() if direction == "stricter" else approve[m].mean()) if n else 0.0
            regions.append(Region(box, direction, n, follows, n >= min_files and follows >= 0.5))
    regions.sort(key=lambda r: (not r.material, -r.n))
    return regions


def prove(policy_node, leaves: list[Leaf], regions: list[Region], typical: dict | None = None) -> bool:
    """Z3: every profile where policy and practice disagree lies in some region.

    Each material region also gets a witness applicant. `typical` (feature ->
    (lo, hi), e.g. the book's 5th-95th percentiles) steers the witness toward a
    believable profile; Z3 falls back to any point in the region if it can't.
    """
    V = logic.z3_vars()
    D = logic.z3_domain(V)
    P = logic.node_z3(policy_node, V)
    A = z3.Or([lf.box.z3(V) for lf in leaves if lf.approve]) if any(lf.approve for lf in leaves) else z3.BoolVal(False)
    covered = z3.Or([r.box.z3(V) for r in regions]) if regions else z3.BoolVal(False)
    complete = logic.solve(D, z3.Xor(P, A), z3.Not(covered)) is None

    for r in regions:
        if not r.material:
            continue
        m = logic.solve(D, r.box.z3(V), *_typical(typical, V)) or logic.solve(D, r.box.z3(V))
        r.witness = logic.witness(m, V) if m is not None else None
        inside_p = logic.solve(D, r.box.z3(V), z3.Not(P)) is None
        outside_p = logic.solve(D, r.box.z3(V), P) is None
        r.z3_label = "entailment" if inside_p else "contradiction" if outside_p else "neutral"
    return complete


def _typical(typical: dict | None, V) -> list:
    out = []
    for f, (lo, hi) in (typical or {}).items():
        if FEATURES[f].kind == "real":
            out += [V[f] >= logic.zr(lo), V[f] <= logic.zr(hi)]
        elif FEATURES[f].kind == "int":
            out += [V[f] >= int(np.ceil(lo)), V[f] <= int(np.floor(hi))]
    return out


# --------------------------------------------------------------------------
# Step 4
# --------------------------------------------------------------------------

@dataclass
class BranchFinding:
    branch: int
    n: int
    residuals: int
    rate: float
    z: float
    direction: str = ""  # "declined-anyway" | "approved-anyway"
    population: int = 0
    rule: tuple | None = None
    metrics: dict | None = None
    twil_rule: tuple | None = None
    twil_metrics: dict | None = None
    twil_raw: str | None = None
    twil_note: str = ""


def branch_residuals(book: pd.DataFrame, predicted: np.ndarray, z_flag: float = 4.0) -> tuple[list[BranchFinding], list[BranchFinding]]:
    """(flagged branches, all branches by descending z)."""
    resid = book["approve"].to_numpy() != predicted
    base = resid.mean()
    out = []
    for b, idx in book.groupby("branch").indices.items():
        n, k = len(idx), int(resid[idx].sum())
        rate = k / n
        z = (rate - base) / np.sqrt(base * (1 - base) / n) if 0 < base < 1 else 0.0
        out.append(BranchFinding(int(b), n, k, rate, float(z)))
    out.sort(key=lambda f: -f.z)
    return [f for f in out if f.z >= z_flag], out


def _metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    tp = int((pred & label).sum())
    prec = tp / pred.sum() if pred.sum() else 0.0
    rec = tp / label.sum() if label.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "flagged": int(pred.sum()), "positives": int(label.sum())}


def branch_population(book: pd.DataFrame, predicted: np.ndarray, finding: BranchFinding):
    """Files in the branch on one side of the practice model, labelled by residual."""
    in_branch = book["branch"].to_numpy() == finding.branch
    approve = book["approve"].to_numpy()
    declined_anyway = int((in_branch & (predicted == 1) & (approve == 0)).sum())
    approved_anyway = int((in_branch & (predicted == 0) & (approve == 1)).sum())
    if declined_anyway >= approved_anyway:
        finding.direction, side = "declined-anyway", in_branch & (predicted == 1)
        label = approve[side] == 0
    else:
        finding.direction, side = "approved-anyway", in_branch & (predicted == 0)
        label = approve[side] == 1
    finding.population = int(side.sum())
    return book[side].reset_index(drop=True), label


def _candidate_atoms(sub: pd.DataFrame):
    atoms = []
    for f in ORDER:
        x = sub[f].to_numpy(dtype=float)
        if FEATURES[f].kind == "bool":
            atoms += [(("cmp", f, "==", 1.0), x == 1), (("cmp", f, "==", 0.0), x == 0)]
            continue
        for t in sorted(set(np.round(np.quantile(x, np.arange(0.05, 0.96, 0.05)), 1))):
            atoms += [(("cmp", f, "<", float(t)), x < t), (("cmp", f, ">=", float(t)), x >= t)]
    return atoms


def search_rule(sub: pd.DataFrame, label: np.ndarray):
    """Best one- or two-condition rule by F1; prefer one condition when it is as good."""
    atoms = _candidate_atoms(sub)
    best1 = max(atoms, key=lambda a: _metrics(a[1], label)["f1"])
    best2, best2_f1 = None, -1.0
    for (n1, m1), (n2, m2) in combinations(atoms, 2):
        if n1[1] == n2[1]:
            continue
        f1 = _metrics(m1 & m2, label)["f1"]
        if f1 > best2_f1:
            best2, best2_f1 = (("and", [n1, n2]), m1 & m2), f1
    m1 = _metrics(best1[1], label)
    if best2 is None or m1["f1"] >= best2_f1 - 0.02:
        return best1[0], m1
    return best2[0], _metrics(best2[1], label)


def validate_rule(node, sub: pd.DataFrame, label: np.ndarray) -> dict:
    return _metrics(logic.node_mask(node, sub), label)

"""The rule language: intervals, boxes, a small formula grammar, DNF, Z3, English.

Everything downstream - the written policy, the decision tree's leaves, the
regions where they disagree, TwIL-LM's proposals - is expressed in one shape:
a *box*, i.e. one interval per feature. A formula is a union of boxes (its
DNF). Keeping a single representation is what lets Z3 prove statements about
the policy and the practice in the same terms the report prints.

Formula grammar (a safe subset of Python expression syntax):

    fico >= 660 and dti <= 45
    self_employed and not existing_customer
    640 <= fico < 660 or (ltv > 95)
"""

from __future__ import annotations

import ast
import math
import re
from fractions import Fraction

import numpy as np
import z3

from .schema import ALIASES, FEATURES, ORDER

INF = math.inf


class RuleError(ValueError):
    """A formula that does not parse, or mentions an unknown attribute."""


# --------------------------------------------------------------------------
# Intervals and boxes
# --------------------------------------------------------------------------

class Interval:
    __slots__ = ("lo", "lo_inc", "hi", "hi_inc")

    def __init__(self, lo=-INF, lo_inc=False, hi=INF, hi_inc=False):
        self.lo, self.lo_inc, self.hi, self.hi_inc = lo, lo_inc, hi, hi_inc

    def key(self):
        return (self.lo, self.lo_inc, self.hi, self.hi_inc)

    def __eq__(self, other):
        return isinstance(other, Interval) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def intersect(self, o: "Interval") -> "Interval":
        if self.lo > o.lo:
            lo, li = self.lo, self.lo_inc
        elif o.lo > self.lo:
            lo, li = o.lo, o.lo_inc
        else:
            lo, li = self.lo, self.lo_inc and o.lo_inc
        if self.hi < o.hi:
            hi, hinc = self.hi, self.hi_inc
        elif o.hi < self.hi:
            hi, hinc = o.hi, o.hi_inc
        else:
            hi, hinc = self.hi, self.hi_inc and o.hi_inc
        return Interval(lo, li, hi, hinc)

    def empty(self) -> bool:
        return self.lo > self.hi or (self.lo == self.hi and not (self.lo_inc and self.hi_inc))


def domain(feat: str) -> Interval:
    f = FEATURES[feat]
    return Interval(f.lo, True, f.hi, True)


def canon(feat: str, iv: Interval) -> Interval:
    """Clip to the feature's domain; integer features get inclusive int bounds.

    After this, `fico > 659.5` and `fico >= 660` are the same Interval, which is
    what makes box equality (and therefore merging) meaningful.
    """
    iv = iv.intersect(domain(feat))
    if FEATURES[feat].kind in ("int", "bool") and not iv.empty():
        lo = math.ceil(iv.lo) if iv.lo_inc else math.floor(iv.lo) + 1
        hi = math.floor(iv.hi) if iv.hi_inc else math.ceil(iv.hi) - 1
        iv = Interval(lo, True, hi, True)
    return iv


class Box:
    """A conjunction of one interval per feature (unconstrained = full domain)."""

    __slots__ = ("ivs",)

    def __init__(self, ivs: dict[str, Interval] | None = None):
        self.ivs = {f: domain(f) for f in ORDER}
        for f, iv in (ivs or {}).items():
            if f not in FEATURES:
                raise RuleError(f"unknown attribute {f!r}")
            self.ivs[f] = canon(f, self.ivs[f].intersect(iv))

    @classmethod
    def _raw(cls, ivs):
        b = cls.__new__(cls)
        b.ivs = ivs
        return b

    def key(self):
        return tuple(self.ivs[f].key() for f in ORDER)

    def empty(self) -> bool:
        return any(iv.empty() for iv in self.ivs.values())

    def intersect(self, other: "Box") -> "Box":
        return Box._raw({f: canon(f, self.ivs[f].intersect(other.ivs[f])) for f in ORDER})

    def constrained(self) -> list[str]:
        return [f for f in ORDER if self.ivs[f] != domain(f)]

    def minus(self, other: "Box") -> list["Box"]:
        """self \\ other as a list of disjoint boxes."""
        if self.intersect(other).empty():
            return [self]
        out, cur = [], dict(self.ivs)
        for f in other.constrained():
            a, b = cur[f], other.ivs[f]
            left = canon(f, a.intersect(Interval(-INF, False, b.lo, not b.lo_inc)))
            right = canon(f, a.intersect(Interval(b.hi, not b.hi_inc, INF, False)))
            for piece in (left, right):
                if not piece.empty():
                    out.append(Box._raw({**cur, f: piece}))
            cur[f] = canon(f, a.intersect(b))
        return out

    def mask(self, df) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for f in self.constrained():
            iv, x = self.ivs[f], df[f].to_numpy(dtype=float)
            m &= (x >= iv.lo) if iv.lo_inc else (x > iv.lo)
            m &= (x <= iv.hi) if iv.hi_inc else (x < iv.hi)
        return m

    def z3(self, V) -> z3.BoolRef:
        parts = []
        for f in self.constrained():
            iv, v = self.ivs[f], V[f]
            lo, hi = (int(iv.lo), int(iv.hi)) if FEATURES[f].kind != "real" else (zr(iv.lo), zr(iv.hi))
            parts.append(v >= lo if iv.lo_inc else v > lo)
            parts.append(v <= hi if iv.hi_inc else v < hi)
        return z3.And(parts) if parts else z3.BoolVal(True)

    def describe(self) -> str:
        parts = []
        for f in self.constrained():
            spec, iv, d = FEATURES[f], self.ivs[f], domain(f)
            if spec.kind == "bool":
                parts.append(spec.label if iv.lo == 1 else f"not {spec.label}")
                continue
            has_lo, has_hi = iv.lo != d.lo or not iv.lo_inc, iv.hi != d.hi or not iv.hi_inc
            if spec.kind == "int" and has_lo and has_hi:
                parts.append(f"{spec.label} {fmt(iv.lo)}–{fmt(iv.hi)}")
            elif has_lo and has_hi:
                parts.append(f"{fmt(iv.lo)} {'≤' if iv.lo_inc else '<'} {spec.label} "
                             f"{'≤' if iv.hi_inc else '<'} {fmt(iv.hi)}")
            elif has_lo:
                parts.append(f"{spec.label} {'≥' if iv.lo_inc else '>'} {fmt(iv.lo)}")
            else:
                parts.append(f"{spec.label} {'≤' if iv.hi_inc else '<'} {fmt(iv.hi)}")
        return ", ".join(parts) or "any applicant"


def fmt(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def disjoint(boxes: list[Box]) -> list[Box]:
    """Rewrite a union of possibly-overlapping boxes as disjoint boxes."""
    out: list[Box] = []
    for b in boxes:
        pieces = [b]
        for o in out:
            pieces = [q for p in pieces for q in p.minus(o)]
        out.extend(p for p in pieces if not p.empty())
    return out


def _touching(feat: str, a: Interval, b: Interval) -> bool:
    if a.lo > b.lo:
        a, b = b, a
    if FEATURES[feat].kind != "real":
        return a.hi + 1 >= b.lo
    return a.hi > b.lo or (a.hi == b.lo and (a.hi_inc or b.lo_inc))


def merge(boxes: list[Box]) -> list[Box]:
    """Fuse boxes that are identical except on one feature, where they touch.

    A decision tree fragments a region along splits that do not matter to it
    (income, loan size); merging puts the region back together for reading.
    """
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                diff = [f for f in ORDER if a.ivs[f] != b.ivs[f]]
                if len(diff) == 1 and _touching(diff[0], a.ivs[diff[0]], b.ivs[diff[0]]):
                    f, x, y = diff[0], a.ivs[diff[0]], b.ivs[diff[0]]
                    lo = min((x.lo, not x.lo_inc), (y.lo, not y.lo_inc))
                    hi = max((x.hi, x.hi_inc), (y.hi, y.hi_inc))
                    boxes[i] = Box._raw({**a.ivs, f: canon(f, Interval(lo[0], not lo[1], hi[0], hi[1]))})
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def union_mask(boxes: list[Box], df) -> np.ndarray:
    m = np.zeros(len(df), dtype=bool)
    for b in boxes:
        m |= b.mask(df)
    return m


# --------------------------------------------------------------------------
# Formula AST:  ("and", [..]) ("or", [..]) ("not", x) ("cmp", f, op, v) ("const", b)
# --------------------------------------------------------------------------

_CMP = {ast.GtE: ">=", ast.LtE: "<=", ast.Gt: ">", ast.Lt: "<", ast.Eq: "==", ast.NotEq: "!="}
_FLIP = {">=": "<=", "<=": ">=", ">": "<", "<": ">", "==": "==", "!=": "!="}
_NEG = {">=": "<", "<=": ">", ">": "<=", "<": ">=", "==": "!=", "!=": "=="}


def parse(text: str):
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        raise RuleError(f"does not parse: {text!r}") from e
    return _conv(tree.body)


def _number(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _number(n.operand)
        return None if v is None else -v
    return None


def _feature(name: str) -> str:
    if name not in FEATURES:
        raise RuleError(f"unknown attribute {name!r}")
    return name


def _conv(n):
    if isinstance(n, ast.BoolOp):
        return ("and" if isinstance(n.op, ast.And) else "or", [_conv(v) for v in n.values])
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
        return ("not", _conv(n.operand))
    if isinstance(n, ast.Name):
        f = _feature(n.id)
        if FEATURES[f].kind != "bool":
            raise RuleError(f"{f!r} is numeric and needs a comparison")
        return ("cmp", f, "==", 1.0)
    if isinstance(n, ast.Constant) and isinstance(n.value, bool):
        return ("const", n.value)
    if isinstance(n, ast.Compare):
        operands, parts = [n.left, *n.comparators], []
        for a, op, b in zip(operands, n.ops, operands[1:]):
            if type(op) not in _CMP:
                raise RuleError(f"unsupported operator {type(op).__name__}")
            sym = _CMP[type(op)]
            if isinstance(a, ast.Name) and _number(b) is not None:
                parts.append(("cmp", _feature(a.id), sym, _number(b)))
            elif isinstance(b, ast.Name) and _number(a) is not None:
                parts.append(("cmp", _feature(b.id), _FLIP[sym], _number(a)))
            else:
                raise RuleError("comparisons must be attribute vs number")
        return parts[0] if len(parts) == 1 else ("and", parts)
    raise RuleError(f"unsupported syntax: {ast.dump(n)[:60]}")


def _atom(op: str, v: float) -> list[Interval]:
    return {
        ">=": [Interval(v, True, INF, False)],
        "<=": [Interval(-INF, False, v, True)],
        ">": [Interval(v, False, INF, False)],
        "<": [Interval(-INF, False, v, False)],
        "==": [Interval(v, True, v, True)],
        "!=": [Interval(-INF, False, v, False), Interval(v, False, INF, False)],
    }[op]


def to_boxes(node, negate: bool = False) -> list[Box]:
    """DNF of a formula (or of its negation) as a list of boxes."""
    kind = node[0]
    if kind == "const":
        return [Box()] if node[1] != negate else []
    if kind == "not":
        return to_boxes(node[1], not negate)
    if kind == "cmp":
        _, f, op, v = node
        out = [Box({f: iv}) for iv in _atom(_NEG[op] if negate else op, v)]
        return [b for b in out if not b.empty()]
    parts = [to_boxes(c, negate) for c in node[1]]
    if (kind == "and") != negate:
        result = [Box()]
        for p in parts:
            result = [x for a in result for b in p if not (x := a.intersect(b)).empty()]
        return result
    return [b for p in parts for b in p]


def node_mask(node, df) -> np.ndarray:
    return union_mask(to_boxes(node), df)


def features_of(node) -> set[str]:
    if node[0] == "cmp":
        return {node[1]}
    if node[0] == "not":
        return features_of(node[1])
    if node[0] in ("and", "or"):
        return set().union(*(features_of(c) for c in node[1]))
    return set()


# --------------------------------------------------------------------------
# Z3
# --------------------------------------------------------------------------

def zr(v: float) -> z3.ArithRef:
    fr = Fraction(v)
    return z3.RealVal(f"{fr.numerator}/{fr.denominator}")


def z3_vars() -> dict[str, z3.ArithRef]:
    return {f: (z3.Real(f) if FEATURES[f].kind == "real" else z3.Int(f)) for f in ORDER}


def z3_domain(V) -> z3.BoolRef:
    parts = []
    for f in ORDER:
        spec = FEATURES[f]
        lo, hi = (zr(spec.lo), zr(spec.hi)) if spec.kind == "real" else (int(spec.lo), int(spec.hi))
        parts += [V[f] >= lo, V[f] <= hi]
    return z3.And(parts)


def node_z3(node, V) -> z3.BoolRef:
    kind = node[0]
    if kind == "const":
        return z3.BoolVal(node[1])
    if kind == "not":
        return z3.Not(node_z3(node[1], V))
    if kind == "and":
        return z3.And([node_z3(c, V) for c in node[1]])
    if kind == "or":
        return z3.Or([node_z3(c, V) for c in node[1]])
    _, f, op, v = node
    lhs = V[f] if FEATURES[f].kind == "real" else z3.ToReal(V[f])
    rhs = zr(v)
    return {">=": lhs >= rhs, "<=": lhs <= rhs, ">": lhs > rhs,
            "<": lhs < rhs, "==": lhs == rhs, "!=": lhs != rhs}[op]


def solve(*constraints, V=None):
    """Return a Z3 model satisfying the constraints within the domain, or None."""
    s = z3.Solver()
    if V is not None:
        s.add(z3_domain(V))
    s.add(*constraints)
    return s.model() if s.check() == z3.sat else None


def witness(model, V) -> dict[str, float]:
    out = {}
    for f in ORDER:
        val = model.eval(V[f], model_completion=True)
        out[f] = val.as_long() if FEATURES[f].kind != "real" else float(val.as_fraction())
    return out


# --------------------------------------------------------------------------
# English (deterministic templates - the round-trip a reviewer reads)
# --------------------------------------------------------------------------

_SYM = {">=": "≥", "<=": "≤", ">": ">", "<": "<", "==": "=", "!=": "≠"}


def english(node) -> str:
    kind = node[0]
    if kind == "const":
        return "always" if node[1] else "never"
    if kind == "not":
        return f"not ({english(node[1])})"
    if kind in ("and", "or"):
        inner = f" {kind} ".join(english(c) for c in node[1])
        return f"({inner})" if kind == "or" else inner
    _, f, op, v = node
    spec = FEATURES[f]
    if spec.kind == "bool":
        positive = (op == "==" and v == 1) or (op == "!=" and v == 0)
        return spec.label if positive else f"not {spec.label}"
    return f"{spec.label} {_SYM[op]} {fmt(v)}"


def to_text(node) -> str:
    """Render a formula back into the grammar `parse` accepts."""
    kind = node[0]
    if kind == "const":
        return str(node[1])
    if kind == "not":
        return f"not ({to_text(node[1])})"
    if kind in ("and", "or"):
        return f" {kind} ".join(f"({to_text(c)})" if c[0] in ("and", "or") else to_text(c) for c in node[1])
    _, f, op, v = node
    if FEATURES[f].kind == "bool":
        return f if (op == "==") == (v == 1) else f"not {f}"
    return f"{f} {op} {fmt(v)}"


# --------------------------------------------------------------------------
# First-order-logic text from TwIL-LM -> the grammar above
# --------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Drop the model's <think> block. An unclosed block means it was truncated."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


_LATEX = [("\\leftrightarrow", "↔"), ("\\iff", "↔"), ("\\rightarrow", "→"), ("\\to", "→"),
          ("\\land", "∧"), ("\\wedge", "∧"), ("\\lor", "∨"), ("\\vee", "∨"),
          ("\\lnot", "¬"), ("\\neg", "¬"), ("\\geq", "≥"), ("\\ge", "≥"),
          ("\\leq", "≤"), ("\\le", "≤"), ("\\neq", "≠")]
# Outcome vocabularies. A policy clause is about approval; an induced rule is
# about "label 1", which may itself be a decline ("declined anyway").
OUTCOMES = {
    "approve": (r"approv", r"declin|reject|den(?:y|ied)"),
    "label": (r"approv|declin|reject|den(?:y|ied)|label|class|positive|override|target", r"$^"),
}


def _outcome(side: str, mode: str) -> str | None:
    """'+' if a formula side is the outcome, '-' if its negation, else None."""
    pos, neg = OUTCOMES[mode]
    if re.search(rf"(¬|\bnot\b|~|!)\s*\(?\s*(?:{pos})", side, re.I) or re.search(neg, side, re.I):
        return "-"
    return "+" if re.search(pos, side, re.I) else None


def _is_guard(side: str) -> bool:
    """A bare unknown predicate such as `Applicant` - it names the domain, not a condition."""
    word = side.strip()
    return bool(re.fullmatch(r"[A-Za-z_]+", word)) and re.sub(r"[^a-z0-9]", "", word.lower()) not in ALIASES


def _balance(s: str) -> str:
    s = s.strip()
    while s.count("(") > s.count(")") and s.startswith("("):
        s = s[1:].strip()
    while s.count(")") > s.count("(") and s.endswith(")"):
        s = s[:-1].strip()
    while s.startswith("(") and s.endswith(")") and _wraps(s):
        s = s[1:-1].strip()
    return s


def _wraps(s: str) -> bool:
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(s) - 1:
            return False
    return True


def _alias(m: re.Match) -> str:
    w = m.group(0)
    if w.lower() in ("and", "or", "not"):
        return w.lower()
    if w in ("True", "False"):
        return w
    return ALIASES.get(re.sub(r"[^a-z0-9]", "", w.lower()), w)


def fol_to_formula(raw: str, mode: str = "approve") -> str:
    """Best-effort conversion of a first-order-logic answer to the rule grammar.

    Handles the shapes small models actually emit - `∀x (Approve(x) → FICO(x) ≥ 660)`,
    LaTeX connectives, `Rule: ...` prefixes. Anything this cannot normalise fails
    the parse gate downstream, which is the intended behaviour: a translation
    nobody can check is a translation nobody uses.
    """
    text = strip_think(raw)
    # Models emit typographic spaces (U+202F narrow no-break, U+00A0, thin
    # spaces) and zero-width joiners between tokens; treat them as plain space.
    text = re.sub(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    text = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", text)
    for a, b in _LATEX:
        text = text.replace(a, b)
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pick = next((ln for ln in reversed(lines) if re.search(r"[<>=≤≥≠]|∧|∨|¬|\band\b|\bor\b", ln)), "")
    s = pick.replace("`", "").replace("$", "").replace("**", "")
    s = re.sub(r"^\s*(?:\d+[.)]\s*)?[A-Za-z ]{0,24}:\s*", "", s)
    s = s.replace("<->", "↔").replace("<=>", "↔").replace("=>", "→").replace("->", "→")
    s = s.replace("\\forall", "∀").replace("\\exists", "∃")
    # Quantifiers are about to be dropped, which is only sound for a universal
    # claim about one applicant. `∃x φ` ("some applicant satisfies φ") states a
    # fact about the population, not a requirement - reject it rather than let
    # it silently become φ. `¬∃x φ` (= ∀x ¬φ) is fine.
    for m in re.finditer(r"∃|\bexists\b", s, flags=re.I):
        if not re.search(r"(¬|~|!|\bnot)\s*$", s[:m.start()], flags=re.I):
            raise RuleError("existential quantifier: says some applicant qualifies, not that every applicant must")
    s = re.sub(r"(∀|∃|\bforall\b|\bexists\b)\s*[a-z]\s*[.:,]?", " ", s, flags=re.I)
    s = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*[a-z]\s*\)", r"\1", s)
    s = _balance(s)

    for sep in ("↔", "→"):
        if sep in s:
            left, right = (_balance(part) for part in s.split(sep, 1))
            lo, ro = _outcome(left, mode), _outcome(right, mode)
            if ro == "+" and lo is None:
                s = left            # cond -> Approve
            elif ro == "-" and lo is None:
                s = f"not ({left})"  # cond -> not Approve / Decline
            elif lo == "+" and ro is None:
                s = right           # Approve -> cond
            elif lo == "-" and ro is None and sep == "↔":
                s = f"not ({right})"
            elif sep == "→" and _is_guard(left):
                s = right           # Applicant(x) -> cond: a domain guard, not an outcome
            break

    for a, b in [("∧", " and "), ("&&", " and "), ("&", " and "), ("∨", " or "), ("||", " or "),
                 ("|", " or "), ("¬", " not "), ("≥", ">="), ("≤", "<="), ("≠", "!=")]:
        s = s.replace(a, b)
    s = re.sub(r"(?<![<>!=])=(?!=)", "==", s)
    s = re.sub(r"~|!(?!=)", " not ", s)
    s = s.replace("%", "")
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    s = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _alias, s)
    return _balance(s.strip().rstrip(".;"))

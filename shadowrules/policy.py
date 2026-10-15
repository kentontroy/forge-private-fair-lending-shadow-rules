"""Step 1 - formalize the written credit policy, clause by clause.

Offline, each clause uses its analyst-approved formula from the policy file.
With TwIL-LM, the model translates each clause to first-order logic and the
translation must clear four gates before anything downstream may use it:

  parse       the answer normalises into the rule grammar
  satisfiable some applicant can pass the clause, and some can fail it
  round-trip  the formula renders back to English for the reviewer to read
  sign-off    equivalent to the analyst's formula (Z3 proves it)

In production the last gate is a human signing off on the round-trip text.
Here it is simulated by an equivalence proof so the demo runs unattended. A
rejected translation falls back to the analyst formula and the report says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import z3

from . import logic


@dataclass
class Clause:
    id: str
    title: str
    text: str
    reason_code: str
    analyst_formula: str


@dataclass
class Policy:
    name: str
    version: str
    decision: str
    clauses: list[Clause]
    reason_codes: dict[str, str]


@dataclass
class Formalization:
    clause: Clause
    node: tuple
    source: str  # "analyst" | "twil" | "analyst (TwIL rejected)"
    gates: dict[str, bool] = field(default_factory=dict)
    raw: str | None = None
    candidate: str | None = None
    note: str = ""

    @property
    def english(self) -> str:
        return logic.english(self.node)


def load_policy(path: Path) -> Policy:
    data = json.loads(Path(path).read_text())
    return Policy(
        name=data["name"], version=data["version"], decision=data["decision"],
        clauses=[Clause(**c) for c in data["clauses"]], reason_codes=data["reason_codes"],
    )


def formalize(policy: Policy, twil=None) -> list[Formalization]:
    V = logic.z3_vars()
    D = logic.z3_domain(V)
    out = []
    for c in policy.clauses:
        ref = logic.parse(c.analyst_formula)
        if twil is None:
            out.append(Formalization(c, ref, "analyst"))
            continue

        raw = twil.translate(c.text)
        gates = {"parse": False, "satisfiable": False, "round-trip": False, "sign-off": False}
        cand_text, note, node = "", "", None
        try:
            cand_text = logic.fol_to_formula(raw)
            node = logic.parse(cand_text)
            gates["parse"] = True
        except logic.RuleError as e:
            note = str(e)
        if node is not None:
            phi = logic.node_z3(node, V)
            gates["satisfiable"] = (logic.solve(D, phi) is not None) and (logic.solve(D, z3.Not(phi)) is not None)
            gates["round-trip"] = bool(logic.english(node))
            ref_z3 = logic.node_z3(ref, V)
            gates["sign-off"] = logic.solve(D, z3.Xor(phi, ref_z3)) is None
            if not gates["sign-off"]:
                note = f"not equivalent to analyst formula; reads as: {logic.english(node)}"
        accepted = all(gates.values())
        out.append(Formalization(
            c, node if accepted else ref, "twil" if accepted else "analyst (TwIL rejected)",
            gates, raw, cand_text, note,
        ))
    return out


def policy_node(forms: list[Formalization]) -> tuple:
    return ("and", [f.node for f in forms])

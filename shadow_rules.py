#!/usr/bin/env python3
"""
Shadow-Policy Audit — rule induction from past lending decisions, for fair-lending review.

The question a fair-lending examiner keeps asking is "does the bank follow its
own written policy, and do the places it doesn't fall harder on one group?"
This demo answers it end to end:

    1  formalize the written policy           analyst formulas, or TwIL-LM + 4 gates
    2  induce the policy as practiced         decision tree on permissible attributes
    3  find where practice departs            box algebra + Z3 completeness proof
    4  explain branch-level overrides          rule search (+ TwIL-LM in-context induction)
    5  measure disparity                      BISG-weighted AIR, counterfactuals, logit
    6  audit adverse-action notices           stated reason vs the clause that fired

The separation of concerns is the point, not the model:

    scikit-learn + Z3 + statsmodels  ->  every finding and every number
    TwIL-LM (optional)               ->  translation only, every answer verified

Usage
-----
    python shadow_rules.py                    # synthetic book, offline (no model)
    python shadow_rules.py --check            # ...and exit 1 unless planted rules are recovered
    python shadow_rules.py --twil             # use TwIL-LM locally (downloads ~1 GB once)
    python shadow_rules.py --twil --model-path ~/models/TwIL-LM2-Q4_K_M.gguf
    python shadow_rules.py --twil --hf-repo webAI-Official/TwIL-LM3-Pro --hf-file TwIL-LM3-Pro-Q8_0.gguf
    python shadow_rules.py --book my_book.csv --demographics bisg.csv

Outputs land in ./out: findings.md (the memo), review_queue.csv (files for
matched-pair review), loan_book.csv (the synthetic book, when generated).

Synthetic data, abstract groups, no legal conclusions: the output is a list of
candidates for counsel and fair-lending statisticians, not a finding of
discrimination.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shadowrules import disparity, induce, logic, notices, policy, synth
from shadowrules.schema import APPLICATION_COLUMNS, DECISION_COLUMNS, FEATURES, ORDER

HERE = Path(__file__).resolve().parent


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def h(self, text: str, level: int = 2):
        self.lines += ["", f"{'#' * level} {text}", ""]

    def p(self, text: str = ""):
        self.lines.append(text)

    def table(self, headers: list[str], rows: list[list]):
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def load_inputs(args):
    if args.book:
        book = pd.read_csv(args.book)
        missing = [c for c in [*ORDER, "app_id", "branch", "approve"] if c not in book.columns]
        if missing:
            sys.exit(f"--book is missing columns: {', '.join(missing)}")
        for c, default in (("underwriter", 0), ("override", 0), ("stated_reason", "")):
            if c not in book.columns:
                book[c] = default
        book["stated_reason"] = book["stated_reason"].fillna("")
        p_b = None
        if args.demographics:
            demo = pd.read_csv(args.demographics)[["app_id", "p_group_b"]]
            p_b = book[["app_id"]].merge(demo, on="app_id", how="left")["p_group_b"].to_numpy(dtype=float)
            if np.isnan(p_b).any():
                sys.exit("--demographics has no p_group_b for some app_id values")
        return book, p_b, None, f"`{args.book}` ({len(book):,} applications)"

    df = synth.generate(n=args.n, seed=args.seed)
    book = df[APPLICATION_COLUMNS + DECISION_COLUMNS].copy()
    book.to_csv(args.out / "loan_book.csv", index=False)
    df[["app_id", "p_group_b"]].to_csv(args.out / "demographics_bisg.csv", index=False)
    return book, df["p_group_b"].to_numpy(), df, f"synthetic, {len(book):,} applications, seed {args.seed}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twil", action="store_true", help="use TwIL-LM for translation, entailment and induction")
    ap.add_argument("--offline", action="store_true", help="no model (the default; kept for symmetry with --twil)")
    ap.add_argument("--model-path", help="local GGUF file; skips the download")
    ap.add_argument("--hf-repo", default=None, help="Hugging Face repo (default webAI-Official/TwIL-LM)")
    ap.add_argument("--hf-file", default=None, help="GGUF filename in the repo (default TwIL-LM2-Q4_K_M.gguf)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="generation budget per answer; 0 (default) = up to the end of the model's context")
    ap.add_argument("--repeat-penalty", type=float, default=1.0,
                    help="penalise tokens repeated within --repeat-window (1.0 = off; 1.1 is mild)")
    ap.add_argument("--repeat-window", type=int, default=64, help="recent tokens the repetition penalty looks back over")
    ap.add_argument("--n-ctx", type=int, default=0,
                    help="llama.cpp context window; 0 (default) = the model's training context")
    ap.add_argument("--policy", type=Path, default=HERE / "data" / "credit_policy.json")
    ap.add_argument("--book", type=Path, help="CSV of decisions to audit instead of the synthetic book")
    ap.add_argument("--demographics", type=Path, help="CSV of app_id,p_group_b (BISG estimates) for --book")
    ap.add_argument("--n", type=int, default=20_000, help="synthetic book size")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-depth", type=int, default=7, help="practice tree depth")
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--min-files", type=int, default=30, help="files a disagreement region needs to be material")
    ap.add_argument("--z-flag", type=float, default=4.0, help="branch residual z-score that flags a branch")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--check", action="store_true", help="exit 1 unless every ground-truth check passes (synthetic only)")
    args = ap.parse_args(argv)
    if args.twil and args.offline:
        ap.error("--twil and --offline are mutually exclusive")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    book, p_b, truth, source = load_inputs(args)
    twil = None
    if args.twil:
        from shadowrules.twil import DEFAULT_FILE, DEFAULT_REPO, TwIL
        print("Loading TwIL-LM ...")
        twil = TwIL(model_path=args.model_path, repo=args.hf_repo or DEFAULT_REPO,
                    filename=args.hf_file or DEFAULT_FILE, n_ctx=args.n_ctx, max_tokens=args.max_tokens,
                    cache_path=args.out / "twil_cache.json",
                    repeat_penalty=args.repeat_penalty, repeat_window=args.repeat_window)

    pol = policy.load_policy(args.policy)
    approve = book["approve"].to_numpy()
    R = Report()
    R.p("# Shadow-Policy Audit — Findings")
    R.p()
    R.p(f"*Book: {source} · approval rate {pct(approve.mean())} · {book['branch'].nunique()} branches · "
        f"mode: {f'TwIL-LM ({twil.model_name}, {twil.runtime}, {twil.context:,}-token context' + (f', repeat penalty {twil.repeat_penalty} over {twil.repeat_window} tokens' if twil.repeat_penalty != 1.0 else '') + ')' if twil else 'offline (no language model)'}*")
    R.p()
    R.p("> Candidates for review by counsel and fair-lending statisticians — not a finding of "
        "discrimination. A practice that departs from policy may have a business justification.")

    # ---- Step 1 ------------------------------------------------------------
    print("Step 1  formalizing the written policy")
    forms = policy.formalize(pol, twil)
    P = policy.policy_node(forms)
    policy_ok = logic.node_mask(P, book).astype(int)
    R.h("1 · The written policy, formalized")
    R.p(f"**{pol.name}** (v{pol.version}). {pol.decision}")
    R.p()
    if twil:
        R.table(["Clause", "Text", "Formula used", "Source", "parse", "satisfiable", "round-trip", "sign-off"],
                [[f.clause.id, f.clause.text, f"`{logic.to_text(f.node)}`", f.source,
                  *(("✓" if f.gates.get(g) else "✗") for g in ("parse", "satisfiable", "round-trip", "sign-off"))]
                 for f in forms])
        for f in forms:
            if f.source != "twil":
                R.p(f"- {f.clause.id}: TwIL-LM answer normalised to `{f.candidate or '∅'}` — {f.note or 'rejected'}.")
        R.p()
    else:
        R.table(["Clause", "Text", "Formula"], [[f.clause.id, f.clause.text, f"`{logic.to_text(f.node)}`"] for f in forms])
    R.p(f"**P(x):** approve when {logic.english(P)}. Written policy approves {pct(policy_ok.mean())} of the book; "
        f"actual approval rate is {pct(approve.mean())}; decisions agree with policy on {pct((policy_ok == approve).mean())} of files.")

    # ---- Step 2 ------------------------------------------------------------
    print("Step 2  inducing the policy as practiced")
    tree, leaves = induce.fit_practice(book, args.max_depth, args.min_leaf)
    pred = induce.practice_predict(tree, book)
    R.h("2 · The policy as practiced, induced")
    R.p(f"A depth-{args.max_depth} decision tree on permissible attributes only ({', '.join(ORDER)}); "
        f"protected-class estimates are not an input. {len(leaves)} leaves; it reproduces "
        f"{pct((pred == approve).mean())} of actual decisions. Largest approve leaves of A(x):")
    R.p()
    top = sorted((lf for lf in leaves if lf.approve), key=lambda lf: -lf.n)[:6]
    R.table(["Files", "Purity", "Practice approves when"], [[lf.n, pct(lf.purity), lf.box.describe()] for lf in top])

    # ---- Step 3 ------------------------------------------------------------
    print("Step 3  locating disagreements and proving completeness with Z3")
    regions = induce.disagreements(P, leaves, book, args.min_files)
    typical = {f: tuple(np.quantile(book[f], [0.05, 0.95])) for f in ORDER if FEATURES[f].kind != "bool"}
    complete = induce.prove(P, leaves, regions, typical)
    material = [r for r in regions if r.material]
    sliver_files = sum(r.n for r in regions if not r.material)
    premise = (f"An applicant is approved if and only if {logic.english(P)}.")
    if twil:
        for r in material:
            r.twil_label, _ = twil.entail(premise, f"An applicant with {r.box.describe()} is approved.")
    R.h("3 · Where practice departs from policy")
    R.p(f"**Z3 completeness proof: {'✓ unsat' if complete else '✗ FAILED'}** — "
        f"{'every' if complete else 'not every'} applicant profile where P(x) ≠ A(x) lies in one of the "
        f"{len(regions)} regions below. {len(material)} are material (≥ {args.min_files} files, decided the practice way "
        f"most of the time); the rest are threshold slivers covering {sliver_files} files.")
    R.p()
    findings = []
    for i, r in enumerate(material, 1):
        name = f"F{i}"
        m = r.box.mask(book)
        findings.append({"name": name, "kind": "region", "mask": m, "region": r,
                         "label": f"{r.headline}: {r.box.describe()}"})
        w = r.witness or {}
        wtxt = ", ".join(witness_text(f, w[f]) for f in ORDER) if w else "—"
        labels = f"Z3 `{r.z3_label}`" + (f" · TwIL-LM `{r.twil_label}` {'✓' if r.twil_label == r.z3_label else '✗'}" if twil else "")
        R.p(f"**{name} · {r.headline}** — {r.box.describe()}")
        R.p(f"- {r.n} files; {pct(r.follows_practice)} decided the practice way. Witness applicant: {wtxt}.")
        R.p(f"- \"Approved under the written policy?\" {labels}")
        R.p()

    # ---- Step 4 ------------------------------------------------------------
    print("Step 4  explaining branch-level overrides")
    flagged, all_branches = induce.branch_residuals(book, pred, args.z_flag)
    base_rate = float((approve != pred).mean())
    for fb in flagged:
        sub, label = induce.branch_population(book, pred, fb)
        fb.rule, fb.metrics = induce.search_rule(sub, label)
        if twil and label.sum() >= 5:
            from shadowrules.twil import sample_examples
            ex, ex_label = sample_examples(sub[ORDER], label)
            fb.twil_raw = twil.induce(ex, ex_label, fb.direction)
            cand = ""
            try:
                cand = logic.fol_to_formula(fb.twil_raw, mode="label")
                fb.twil_rule = logic.parse(cand)
                fb.twil_metrics = induce.validate_rule(fb.twil_rule, sub, label)
            except logic.RuleError as e:
                fb.twil_note = f"unparseable (`{cand or '∅'}`): {e}"
        rule = fb.rule
        if fb.twil_metrics and fb.twil_metrics["f1"] >= max(0.5, fb.metrics["f1"] - 0.05):
            rule = fb.twil_rule
        bmask = (book["branch"].to_numpy() == fb.branch) & logic.node_mask(rule, book)
        findings.append({"name": f"F{len(findings) + 1}", "kind": "branch", "mask": bmask, "branch": fb,
                         "rule_used": rule, "label": f"Branch {fb.branch} {fb.direction}: {logic.english(rule)}"})
    R.h("4 · Overrides the practice model cannot explain")
    R.p(f"Residual = a file decided against A(x). Book-wide residual rate {pct(base_rate)}; "
        f"a branch is flagged at z ≥ {args.z_flag:g}.")
    R.p()
    R.table(["Branch", "Files", "Residuals", "Rate", "z"],
            [[b.branch, b.n, b.residuals, pct(b.rate), f"{b.z:+.1f}" + (" **flagged**" if b in flagged else "")]
             for b in all_branches[:5]])
    if not flagged:
        R.p("No branch is flagged.")
    for f in findings:
        if f["kind"] != "branch":
            continue
        fb, mt = f["branch"], f["branch"].metrics
        R.p(f"**{f['name']} · Branch {fb.branch}, {fb.direction}** ({fb.population} files the practice model "
            f"expected to {'approve' if fb.direction == 'declined-anyway' else 'decline'})")
        R.p(f"- Rule search: `{logic.to_text(fb.rule)}` — {logic.english(fb.rule)}; precision {pct(mt['precision'])}, "
            f"recall {pct(mt['recall'])}, F1 {mt['f1']:.2f}.")
        if twil:
            if fb.twil_metrics:
                tm = fb.twil_metrics
                R.p(f"- TwIL-LM in-context induction (40 files shown): `{logic.to_text(fb.twil_rule)}` — validated on all "
                    f"{fb.population} files: precision {pct(tm['precision'])}, recall {pct(tm['recall'])}, F1 {tm['f1']:.2f}"
                    f"{' — **used**' if f['rule_used'] is fb.twil_rule else ' — not used (rule search validated better)'}.")
            else:
                R.p(f"- TwIL-LM in-context induction: {fb.twil_note or 'no answer'} — rule search used.")
        R.p()

    # ---- Step 5 ------------------------------------------------------------
    rule_feats = sorted({f for x in findings for f in (x["region"].box.constrained() if x["kind"] == "region"
                                                        else logic.features_of(x["rule_used"]))},
                        key=ORDER.index)
    policy_feats = set().union(*(logic.features_of(f.node) for f in forms))
    R.h("5 · Disparity (BISG-weighted)")
    if p_b is None:
        print("Step 5  skipped (no demographic estimates)")
        R.p("Skipped — no `--demographics` file supplied.")
    else:
        print("Step 5  measuring disparity")
        a0, ra, rb = disparity.air(approve, p_b)
        lo, hi = disparity.air_ci(approve, p_b)
        a_pol = disparity.air(policy_ok, p_b)[0]
        R.p(f"Approval rate group A {pct(ra)}, group B {pct(rb)} → **AIR {a0:.3f}** (95% bootstrap CI {lo:.3f}–{hi:.3f}). "
            f"Had every file been decided by the written policy, AIR would be {a_pol:.3f}.")
        R.p()
        rows = []
        for f in findings:
            f["cf_air"] = disparity.counterfactual_air(approve, p_b, f["mask"], policy_ok)
            rows.append([f["name"], f["label"], int(f["mask"].sum()), pct(disparity.group_share(p_b, f["mask"])),
                         f"{f['cf_air']:.3f} ({f['cf_air'] - a0:+.3f})"])
        R.p(f"Book-wide estimated group-B share: {pct(p_b.mean())}.")
        R.p()
        R.table(["Finding", "Rule", "Files", "Est. group-B share", "AIR if decided by policy"], rows)
        base = disparity.odds_ratio(book, p_b, policy_ok, [])
        extra = [f for f in rule_feats if f not in policy_feats]
        full = disparity.odds_ratio(book, p_b, policy_ok, extra) if extra else None
        R.p("Logistic regression, odds ratio of approval for p_group_b = 1 vs 0:")
        R.p(f"- controlling for the written-policy factors: **OR {base['or']:.2f}** (95% CI {base['lo']:.2f}–{base['hi']:.2f})")
        if full:
            R.p(f"- also controlling for {', '.join(extra)}: **OR {full['or']:.2f}** "
                f"(95% CI {full['lo']:.2f}–{full['hi']:.2f}). The factors that absorb the disparity are the ones whose "
                "business justification counsel needs to examine.")
        R.p()
        R.p("Proxy scan — correlation of each attribute the findings use with p_group_b:")
        R.p()
        R.table(["Attribute", "r", "In written policy?", "Flag"],
                [[FEATURES[f].label, f"{r:+.2f}", "yes" if f in policy_feats else "**no**",
                  "proxy review" if abs(r) >= 0.10 and f not in policy_feats else ""]
                 for f, r in disparity.proxy_scan(book, p_b, rule_feats)])

    # ---- Step 6 ------------------------------------------------------------
    print("Step 6  auditing adverse-action notices")
    status = notices.classify(book, forms)
    declined = approve == 0
    R.h("6 · Adverse-action notices")
    counts = pd.Series(status[declined]).value_counts()
    R.table(["Status", "Notices", "Share"],
            [[s, int(counts.get(s, 0)), pct(counts.get(s, 0) / declined.sum())]
             for s in ("consistent", "inaccurate", "off-policy", "missing") if counts.get(s, 0)])
    rows = []
    for f in findings:
        m = f["mask"] & declined
        bad = m & np.isin(status, ["inaccurate", "off-policy"])
        top_reason = pd.Series(book["stated_reason"].to_numpy()[bad]).value_counts()
        rows.append([f["name"], int(m.sum()), int(bad.sum()), pct(bad.sum() / m.sum()) if m.sum() else "—",
                     ", ".join(f"{k} ({v})" for k, v in top_reason.head(2).items()) or "—"])
    R.table(["Finding", "Declines", "Inaccurate or off-policy", "Share", "Most-cited reasons"], rows)

    # ---- Review queue ------------------------------------------------------
    queue = []
    for f in findings:
        m = f["mask"] & (approve != policy_ok)
        sub = book[m].assign(finding=f["name"], policy_outcome=policy_ok[m], notice_status=status[m])
        if p_b is not None:
            sub = sub.assign(p_group_b=p_b[m]).sort_values("p_group_b", ascending=False)
        queue.append(sub)
    if queue:
        q = pd.concat(queue)
        cols = ["finding", "app_id", "branch", "underwriter", *ORDER, "approve", "policy_outcome",
                "stated_reason", "notice_status"] + (["p_group_b"] if p_b is not None else [])
        q[cols].to_csv(args.out / "review_queue.csv", index=False)
        R.h("Files for matched-pair review")
        R.p(f"`{args.out / 'review_queue.csv'}` — {len(q):,} files decided against the written policy inside a finding, "
            "highest estimated group-B probability first within each finding.")

    # ---- Ground truth ------------------------------------------------------
    checks = []
    if truth is not None:
        checks = ground_truth(R, truth, book, findings, flagged, complete, status, p_b, approve, forms, twil, material)

    elapsed = time.time() - t0
    if twil:
        write_summary(args, twil, forms, material, findings, checks, elapsed)
    R.p()
    R.p(f"*Generated in {elapsed:.1f}s by `shadow_rules.py`.*")
    (args.out / "findings.md").write_text(R.text())
    print()
    print(R.text())
    print(f"Wrote {args.out / 'findings.md'}" + (f" and {args.out / 'review_queue.csv'}" if queue else ""))

    if args.check:
        if truth is None:
            print("--check needs the synthetic book (no ground truth for --book)")
            return 2
        failed = [name for name, passed in checks if not passed]
        print("ground truth: all checks pass" if not failed else f"ground truth FAILED: {', '.join(failed)}")
        return 1 if failed else 0
    return 0


def witness_text(f: str, v: float) -> str:
    spec = FEATURES[f]
    if spec.kind == "bool":
        return spec.label if v == 1 else f"not {spec.label}"
    return f"{spec.label} {logic.fmt(v)}"


def _bound(node, feat):
    """(op, value) pairs a rule places on one feature."""
    if node[0] == "cmp":
        return [(node[2], node[3])] if node[1] == feat else []
    if node[0] in ("and", "or"):
        return [b for c in node[1] for b in _bound(c, feat)]
    return _bound(node[1], feat) if node[0] == "not" else []


def ground_truth(R, truth, book, findings, flagged, complete, status, p_b, approve, forms, twil, material):
    R.h("Ground truth (synthetic book only)")
    R.p("The generator planted three shadow rules (see `shadowrules/synth.py`). Did the pipeline recover them?")
    R.p()
    regions = [f for f in findings if f["kind"] == "region"]

    def find_r1(f):
        b = f["region"].box.ivs
        return f["region"].direction == "stricter" and b["self_employed"].lo == 1 and 36.5 <= b["dti"].lo <= 39.5

    def find_r2(f):
        b = f["region"].box.ivs
        return (f["region"].direction == "looser" and b["existing_customer"].lo == 1
                and 635 <= b["fico"].lo <= 645 and b["fico"].hi <= 659)

    def find_r3(f):
        if f["kind"] != "branch" or f["branch"].branch != 7:
            return False
        return any(op in ("<", "<=") and 1.0 <= v <= 3.0 for op, v in _bound(f["rule_used"], "years_at_address"))

    r1 = next((f for f in regions if find_r1(f)), None)
    r2 = next((f for f in regions if find_r2(f)), None)
    r3 = next((f for f in findings if find_r3(f)), None)
    false_branches = [b.branch for b in flagged if b.branch != 7]

    gt_bad = truth["_gt_inaccurate_notice"].to_numpy() == 1
    gt_off = truth["_gt_offpolicy_notice"].to_numpy() == 1
    pure_policy = (truth["_gt_rule"].to_numpy() == "") & (approve == 0)
    checks = [
        ("R1 recovered (self-employed, DTI > 38, stricter)", r1 is not None),
        ("R2 recovered (existing customer, FICO 640–659, looser)", r2 is not None),
        ("R3 recovered (branch 7, < 2 years at address)", r3 is not None),
        ("No other branch falsely flagged", not false_branches),
        ("Z3 completeness proof", complete),
        ("Planted inaccurate notices caught (≥ 95%)", (status[gt_bad] == "inaccurate").mean() >= 0.95),
        ("Planted off-policy notices caught (≥ 95%)", (status[gt_off] == "off-policy").mean() >= 0.95),
        ("No false alarms on pure policy declines", not np.isin(status[pure_policy], ["inaccurate", "off-policy"]).any()),
    ]
    if p_b is not None:
        a0 = disparity.air(approve, p_b)[0]
        lifts = [f for f in (r1, r2, r3) if f is not None]
        checks.append(("Each recovered rule's counterfactual raises AIR",
                       bool(lifts) and all(f["cf_air"] > a0 for f in lifts)))
    R.table(["Check", "Result"], [[name, ok(passed)] for name, passed in checks])

    if p_b is not None:
        true_b = (truth["_true_group"].to_numpy() == "B").astype(float)
        a_true, a_est = disparity.air(approve, true_b)[0], disparity.air(approve, p_b)[0]
        R.p(f"BISG error bars, made visible: AIR with the *true* group labels is {a_true:.3f}; with BISG-style "
            f"estimates it is {a_est:.3f}. Probability weighting attenuates the measured gap toward 1.")
    if twil:
        agree = sum(r.twil_label == r.z3_label for r in material)
        accepted = sum(f.source == "twil" for f in forms)
        R.p()
        R.p(f"TwIL-LM: {accepted}/{len(forms)} clause translations passed all gates; entailment labels agreed with "
            f"Z3 on {agree}/{len(material)} regions; {twil.calls} model calls this run (the rest came from cache).")
        st = twil.summary()
        if st["tokens"]:
            R.p(f"Generation: {st['tokens']:,} tokens over {st['answers']} answers in {st['seconds']:,}s "
                f"({st['tok_per_s']} tok/s); longest answer {st['max_tokens_one_answer']:,} tokens; "
                f"{st['length_stops']} answer(s) stopped by the length limit.")
    return checks


def write_summary(args, twil, forms, material, findings, checks, elapsed):
    """Machine-readable result of a model run, for compare_models.py."""
    induction = []
    for f in findings:
        if f["kind"] != "branch":
            continue
        fb = f["branch"]
        if fb.twil_metrics is None:
            status = "unparseable"
        elif f["rule_used"] is fb.twil_rule:
            status = "used"
        else:
            status = "validated, not used"
        induction.append({"branch": fb.branch, "status": status, "rule": logic.to_text(fb.twil_rule) if fb.twil_rule else None,
                          "f1": round(fb.twil_metrics["f1"], 3) if fb.twil_metrics else None,
                          "baseline_f1": round(fb.metrics["f1"], 3)})
    out = {
        "model": twil.model_name, "runtime": twil.runtime, "context": twil.context,
        "max_tokens": args.max_tokens or "model max",
        "repeat_penalty": args.repeat_penalty, "repeat_window": args.repeat_window,
        "clauses": {f.clause.id: {"accepted": f.source == "twil", "candidate": f.candidate, "note": f.note} for f in forms},
        "entailment": [{"region": f"F{i}", "z3": r.z3_label, "twil": r.twil_label} for i, r in enumerate(material, 1)],
        "induction": induction,
        "ground_truth_pass": all(passed for _, passed in checks) if checks else None,
        "elapsed_s": round(elapsed, 1),
        "generation": twil.summary(),
    }
    (args.out / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())

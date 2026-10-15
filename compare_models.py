#!/usr/bin/env python3
"""
Run the audit under several TwIL-LM builds and tabulate how each did.

Each configuration runs `shadow_rules.py --twil --check` into its own
out/compare/<name>/ directory (answers are cached there, so a re-run is
instant), then this script reads every summary.json and writes
out/compare/comparison.md.

No token caps by default: each model gets its full training context and may
generate to the end of it. Runs are sequential so they don't contend for the
GPU. The TwIL-LM3-Pro configurations use a mild repetition penalty (1.1 over
the last 64 tokens) because greedy decoding put the Q4 build into a loop that
only the context limit would end; the original TwIL-LM runs without one, as
its earlier results were measured.

    python compare_models.py                      # all three
    python compare_models.py --only lm3-q4 lm3-q8
    python compare_models.py --table-only         # re-tabulate existing runs
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIGS = {
    "lm2-q4": dict(label="TwIL-LM (SmolLM2 1.7B) · Q4_K_M · llama.cpp",
                   repo="webAI-Official/TwIL-LM", file="TwIL-LM2-Q4_K_M.gguf", repeat_penalty=1.0),
    "lm3-q4": dict(label="TwIL-LM3-Pro (Granite 3.7B) · Q4_K_M · llama.cpp",
                   repo="webAI-Official/TwIL-LM3-Pro", file="TwIL-LM3-Pro-Q4_K_M.gguf", repeat_penalty=1.1),
    "lm3-q8": dict(label="TwIL-LM3-Pro · Q8_0 · llama.cpp",
                   repo="webAI-Official/TwIL-LM3-Pro", file="TwIL-LM3-Pro-Q8_0.gguf", repeat_penalty=1.1),
}


def run(name: str, cfg: dict, out: Path, max_tokens: int) -> None:
    cmd = [sys.executable, str(HERE / "shadow_rules.py"), "--twil", "--check",
           "--hf-repo", cfg["repo"], "--hf-file", cfg["file"], "--max-tokens", str(max_tokens),
           "--out", str(out), "--repeat-penalty", str(cfg["repeat_penalty"])]
    print(f"\n=== {name}: {cfg['label']}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out / "run.log").write_text(proc.stdout + proc.stderr)
    tail = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"    exit {proc.returncode} in {time.time() - t0:.0f}s — {tail}", flush=True)


def cell_clause(c: dict) -> str:
    if c["accepted"]:
        return "✓"
    note = (c["note"] or "rejected").split(";")[0]
    return f"✗ {note[:48]}"


def table(names: list[str], root: Path) -> str:
    cols = [(n, json.loads((root / n / "summary.json").read_text())) for n in names if (root / n / "summary.json").exists()]
    if not cols:
        return "No completed runs."
    head = ["", *[CONFIGS[n]["label"] for n, _ in cols]]
    rows = []

    def row(label, fn):
        rows.append([label, *[fn(s) for _, s in cols]])

    row("Context window (tokens)", lambda s: f"{s['context']:,}")
    row("Repetition penalty", lambda s: f"{s['repeat_penalty']} / last {s['repeat_window']}" if s["repeat_penalty"] != 1.0 else "off")
    for cid in cols[0][1]["clauses"]:
        row(f"Clause {cid}", lambda s, cid=cid: cell_clause(s["clauses"][cid]))
    row("**Clauses passing every gate**", lambda s: f"**{sum(c['accepted'] for c in s['clauses'].values())}/{len(s['clauses'])}**")
    row("**Entailment labels matching Z3**",
        lambda s: f"**{sum(e['twil'] == e['z3'] for e in s['entailment'])}/{len(s['entailment'])}**")
    row("Branch rule induction", lambda s: "; ".join(
        f"{i['status']}" + (f" (F1 {i['f1']:.2f} vs search {i['baseline_f1']:.2f})" if i["f1"] is not None else "")
        for i in s["induction"]) or "—")
    row("Ground-truth checks", lambda s: "all pass" if s["ground_truth_pass"] else "FAIL")
    row("Tokens generated", lambda s: f"{s['generation']['tokens']:,}")
    row("Longest single answer (tokens)", lambda s: f"{s['generation']['max_tokens_one_answer']:,}")
    row("Answers stopped by length", lambda s: str(s["generation"]["length_stops"]))
    row("Generation speed", lambda s: f"{s['generation']['tok_per_s']} tok/s" if s["generation"]["tok_per_s"] else "—")
    row("Model time", lambda s: f"{s['generation']['seconds']:,.0f} s")

    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=list(CONFIGS), help="run a subset")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 (default) = up to the end of the model's context")
    ap.add_argument("--out", type=Path, default=Path("out/compare"))
    ap.add_argument("--table-only", action="store_true", help="skip running; tabulate existing summaries")
    args = ap.parse_args()
    names = args.only or list(CONFIGS)
    if not args.table_only:
        for n in names:
            (args.out / n).mkdir(parents=True, exist_ok=True)
            run(n, CONFIGS[n], args.out / n, args.max_tokens)
    md = table(names, args.out)
    hw = f"{platform.machine()} · {platform.system()} {platform.release()}"
    (args.out / "comparison.md").write_text(f"# TwIL-LM build comparison\n\n*{hw} · greedy decoding · "
                                            f"max tokens: {args.max_tokens or 'model max'}*\n\n{md}\n")
    print("\n" + md + f"\n\nWrote {args.out / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Shadow-Policy Audit

Rule induction from past lending decisions, for fair-lending review — with
TwIL-LM (webAI's 1.7B formal-logic model) as an optional, fully-verified
translator.

Fair-lending exams rarely find discrimination in the written credit policy.
They find it in practice: overrides, exceptions, branch habits nobody wrote
down. This demo recovers **the policy as practiced** from the decision
history, proves exactly where it departs from **the policy as written**, and
measures whether those departures fall harder on one group.

```
1  formalize the written policy      analyst formulas, or TwIL-LM + 4 gates
2  induce the policy as practiced    decision tree, permissible attributes only
3  find where practice departs       box algebra + Z3 completeness proof
4  explain branch-level overrides    rule search (+ TwIL-LM in-context induction)
5  measure disparity                 BISG-weighted AIR, counterfactuals, logit
6  audit adverse-action notices      stated reason vs the clause that fired
```

**The separation of concerns is the point.** scikit-learn, Z3 and statsmodels
produce every finding and every number. TwIL-LM only translates — policy text
to logic, regions to entailment labels, example files to a candidate rule —
and each answer is checked by something deterministic before it is used. The
offline mode runs the same pipeline with no model at all.

## Quick start

Python 3.10+. `requirements.txt` covers the offline pipeline; `requirements-twil.txt`
adds a local model through llama.cpp.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python shadow_rules.py              # synthetic 20k-file book, ~6 s
python shadow_rules.py --check      # exit 1 unless the planted rules are recovered
```

Outputs, in `./out/`:

| File | What it is |
|---|---|
| `findings.md` | The memo: every finding, in English and in logic, with counts, disparity statistics and notice audit |
| `review_queue.csv` | Files decided against the written policy inside a finding, highest estimated group-B probability first — the pull list for matched-pair file review |
| `loan_book.csv`, `demographics_bisg.csv` | The synthetic book and its BISG-style estimates, kept as separate tables on purpose |
| `twil_cache.json`, `twil_progress.log` | TwIL-LM answers (cached by model, budget, penalty and prompt) and live generation progress (`--twil` only) |
| `summary.json` | Machine-readable model scores for the run (`--twil` only) |

## With TwIL-LM

The model runs through llama.cpp from a GGUF file, which embeds the repo's
`chat_template.jinja` verbatim:

```bash
# llama.cpp + GGUF — Metal, CUDA or CPU
pip install -r requirements-twil.txt
python shadow_rules.py --twil                                   # TwIL-LM2 Q4_K_M (~1 GB)
python shadow_rules.py --twil --hf-repo webAI-Official/TwIL-LM3-Pro \
       --hf-file TwIL-LM3-Pro-Q8_0.gguf --repeat-penalty 1.1   # recommended
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-tokens` | `0` | Generation budget per answer; `0` = run to the end of the model's context |
| `--n-ctx` | `0` | llama.cpp context window; `0` = the model's training context (8,192 for TwIL-LM2, 131,072 for TwIL-LM3-Pro) |
| `--repeat-penalty` / `--repeat-window` | `1.0` / `64` | Penalise tokens repeated in the last 64; `1.0` = off |

Decoding is greedy, per the model cards. The defaults assume a large machine:
TwIL-LM3-Pro's full 131K context needs roughly 21 GB of KV cache on top of the
weights. On a 16 GB GPU, pass `--n-ctx 32768` or similar. Answers are cached in
`out/twil_cache.json`, keyed by model, budget, penalty and prompt, so a re-run
makes no model calls. Each answer also streams progress to
`out/twil_progress.log` — `tail -f` it to see tokens, speed and the last 80
characters, which makes a repetition loop obvious while it happens.

What the model does, and what checks it:

| Job | TwIL-LM produces | Checked by |
|---|---|---|
| Formalize each policy clause | first-order logic | parse → satisfiable (Z3) → English round-trip → sign-off (Z3 proves equivalence to the analyst formula) |
| Label each disagreement region | `entailment` / `contradiction` / `neutral` | Z3's own label for the same question |
| Explain a flagged branch | a rule, from 40 labelled files in-context | precision/recall on *every* file in that branch |

A translation that fails any gate falls back to the analyst formula, and the
memo says so. A proposed rule is used only if it validates at least as well as
the exhaustive rule search. In production the sign-off gate is a person reading
the round-trip; here it is simulated by an equivalence proof so the demo runs
unattended.

### Model comparison (default seed, 2026-10-14)

Apple M3 Ultra, 96 GB unified memory. Greedy decoding, no token caps (each
model may generate to the end of its context). Reproduce with
`python compare_models.py`; it writes `out/compare/comparison.md`.

| | TwIL-LM · Q4_K_M · llama.cpp | TwIL-LM3-Pro · Q4_K_M · llama.cpp | TwIL-LM3-Pro · Q8_0 · llama.cpp |
|---|---|---|---|
| Base model | SmolLM2 1.7B | Granite 4.2 3B (3.66B params) | same |
| Weights on disk | 0.98 GiB | 2.09 GiB | 3.63 GiB |
| Context window | 8,192 | 131,072 | 131,072 |
| Repetition penalty | off | 1.1 / last 64 | 1.1 / last 64 |
| **Clauses passing every gate** | **1/3** | **3/3** | **3/3** |
| **Entailment matching Z3** | **1/3** | **2/3** | **3/3** |
| Branch-7 rule induction | unparseable | **used** — `years_at_address <= 2`, F1 0.92 (search 0.92) | validated, not used — F1 0.83 |
| Ground-truth checks | all pass | all pass | all pass |
| Tokens generated (7 answers) | 740 | 8,482 | 32,106 |
| Longest single answer | 238 | 4,467 | 24,092 |
| Model time | 5 s | 113 s | 20 min |

What the comparison shows:

- **TwIL-LM3-Pro is a different class of model from TwIL-LM.** With a
  repetition penalty, every Pro build translated all three clauses correctly,
  against 1/3 for the original. TwIL-LM's misses were substantive: it read the
  45% DTI clause as "FICO + DTI + LTV + … ≤ 0.45".
- **The repetition penalty mattered more than quantization.** In an earlier
  run without it (and with a 4,096-token cap), Pro Q4 managed 2/3 clauses, its induction answer looped, and it spent about 3,000
  tokens per clause reasoning. With it, the same weights scored 3/3, recovered
  the planted branch-7 rule, and reasoned in 800–1,300 tokens per clause.
- **More precision did not mean better answers.** Q4 and Q8_0 tied on
  clauses. Only Q8_0 got all three entailment labels right, and only Q4
  produced a usable rule. Seven questions on one book cannot rank
  quantizations; they show the variance is between runs, not between
  precisions.
- **Rule induction is the model's weak job.** One build out of three produced a
  rule as good as the exhaustive search. Exhaustive search should do induction,
  with the model as an optional second opinion.
- **Uncapped reasoning has a cost.** One Q8_0 answer ran 24,092 tokens, and
  that build's seven answers took 20 minutes. Average speed figures are
  dominated by long answers (attention slows as the context grows), so they aren't a fair
  throughput benchmark. In production, cap the budget well below the context
  and treat "stopped by length" as a failed answer.
- **The gates caught real errors, including mine.** They rejected TwIL-LM's
  wrong clauses and an earlier Pro answer that invented an existing-customer
  condition. They also exposed two normalizer bugs, both fixed: an
  existential `∃x (FICO(x) ≥ 660)` used to slip through because quantifiers
  were dropped before Z3 saw the formula, and Unicode narrow no-break spaces
  (U+202F) made a correct `fico >= 660` unparseable.

One thing observed about the models themselves. TwIL-LM's official chat
template (in the repo and the GGUF alike) injects SmolLM's default "helpful AI
assistant" system prompt, and it produced no `<think>` block. The
prompts in `shadowrules/twil.py` are deliberately plain and have not been tuned
against this book.

Both models are released under the **webAI Non-Commercial License v1.0** —
fine for this demo; production use needs a commercial agreement with webAI. If
the files move, pass `--hf-repo` / `--hf-file` or `--model-path`.

## The synthetic book and its ground truth

`shadowrules/synth.py` writes a 20,000-application book. The written policy
(`data/credit_policy.json`) approves when FICO ≥ 660, DTI ≤ 45 and LTV ≤ 95.
Practice departs from it in three planted ways:

| | Rule | Scope | Direction |
|---|---|---|---|
| R1 | Self-employed applicants above 38% DTI declined | book-wide, undocumented | stricter |
| R2 | Existing customers with FICO 640–659 approved | book-wide overrides | looser |
| R3 | Under two years at address declined | branch 7 overrides | stricter |

Groups are abstract (A, B). Group B is drawn as more often self-employed, less
often an existing customer, with shorter residence and a slightly lower FICO
mean — so every planted rule widens the approval gap, and part of the gap is a
legitimate credit factor. The pipeline sees only a noisy BISG-style
probability, `p_group_b`; the true group is used for scoring alone. Stated
adverse-action reasons are planted too (R1 declines mostly cite DTI, though the
applicant's DTI is within policy).

`--check` scores the run: R1–R3 recovered, no false branch flags, Z3
completeness, ≥ 95% of planted bad notices caught, zero false alarms on pure
policy declines, and each recovered rule's counterfactual raising the AIR. It
passes on seeds 1, 2, 3, 7, 11 and 42, and at `--n 8000`.

## Auditing your own decisions

```bash
python shadow_rules.py --book decisions.csv --demographics bisg.csv
python shadow_rules.py --book decisions.csv --policy my_policy.json
```

`decisions.csv` needs `app_id`, `branch`, `approve` (0/1) and the eight
attributes in `shadowrules/schema.py` (`fico`, `dti`, `ltv`, `income`,
`loan_amount`, `years_at_address`, `self_employed`, `existing_customer`);
`underwriter`, `override` and `stated_reason` are optional. `bisg.csv` is
`app_id,p_group_b`; without it step 5 is skipped. To audit different
attributes, edit `schema.py` — every formula, the tree and the Z3 domain read
from it. A policy file is a list of clauses, each with its text, an analyst
formula in the rule grammar (`fico >= 660 and not self_employed`) and the
reason code a notice would cite.

## Reading the output honestly

- **Disagreement is not discrimination.** A stricter rule for self-employed
  applicants may be justified by income volatility. The memo lists candidates;
  counsel and fair-lending statisticians decide.
- **The regression tells you where to look.** The odds ratio on `p_group_b`
  is computed twice: controlling for the written-policy factors, then also
  for the practice factors the findings use. The factors that absorb the
  disparity are the ones whose business justification needs examining.
- **Proxies.** Attributes outside the written policy that correlate with the
  group estimate are flagged for proxy review even though protected
  characteristics never entered the model.
- **BISG has error bars.** The ground-truth section prints the AIR under true
  labels next to the estimated one; probability weighting pulls the measured
  gap toward 1. Report intervals, not points.
- **Threshold slivers.** Tree split points rarely land exactly on policy
  thresholds; Z3 accounts for every sliver in the completeness proof, and the
  memo reports how many files they hold (usually zero).

## Layout

```
shadow_rules.py          CLI and memo writer
compare_models.py        run several TwIL-LM builds and tabulate them
shadowrules/
  schema.py              the attributes and their domains
  logic.py               intervals, boxes, rule grammar, DNF, Z3, English, FOL normaliser
  synth.py               synthetic book with planted shadow rules
  policy.py              step 1: formalization and gates
  induce.py              steps 2-4: practice tree, disagreement regions, proof, branch rules
  disparity.py           step 5: AIR, counterfactuals, logit, proxy scan
  notices.py             step 6: adverse-action consistency
  twil.py                optional TwIL-LM wrapper (llama.cpp), progress log
data/credit_policy.json  the written policy, with analyst formulas
```

Synthetic data, abstract groups, no legal advice.

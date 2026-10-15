"""TwIL-LM, run locally - optional, through llama.cpp and a GGUF file.

The model never makes a finding. It does three translation jobs, and every
answer is checked by something deterministic before it is used:

  translate  policy clause -> first-order logic     (checked by the policy gates)
  entail     policy + region -> entailment label    (checked against Z3)
  induce     ~40 labelled files -> a rule           (checked against the full branch)

Decoding is greedy, per the model cards, with an optional repetition penalty
over a sliding window of recent tokens (`repeat_penalty`, `repeat_window`):
greedy decoding on a small model can fall into a loop that only the context
limit ends. Keep it mild - formulas legitimately repeat `(x)`, connectives and
attribute names. By default nothing caps the answer
below what the model supports: the context window is the model's training
context and generation may run to its end (`n_ctx=0`, `max_tokens=0`). Lower
both on a small machine. Answers are cached on disk keyed by model, budget and
prompt, so re-running the demo is instant and reproducible.

Runtime: llama-cpp-python + a GGUF file (Q4_K_M ... Q8_0), on Metal, CUDA or
CPU. The GGUF embeds the repo's chat template verbatim.

The models are released under the webAI Non-Commercial License v1.0 - fine for
this demo, not for production use without a commercial agreement.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import logic
from .schema import FEATURES, ORDER

DEFAULT_REPO = "webAI-Official/TwIL-LM"
DEFAULT_FILE = "TwIL-LM2-Q4_K_M.gguf"

class Progress:
    """Live progress for one answer, appended to a log you can `tail -f`.

    A line every `every` tokens (and at the end): tokens so far, speed, and the
    last 80 characters generated - a repetition loop is obvious at a glance.
    """

    def __init__(self, path: Path | None, job: str, budget: int, every: int = 1000):
        self.path, self.job, self.budget, self.every = path, job, budget, every
        self.t0, self.n, self.tail = time.time(), 0, ""

    def step(self, piece: str):
        self.n += 1
        self.tail = (self.tail + piece)[-80:]
        if self.n % self.every == 0:
            self._write("…")

    def done(self, finish: str):
        self._write(f"done ({finish})")

    def _write(self, state: str):
        dt = time.time() - self.t0
        line = (f"{time.strftime('%H:%M:%S')}  {self.job:<9} {self.n:>7,} / {self.budget:,} tokens  "
                f"{self.n / dt if dt else 0:5.1f} tok/s  {state}  tail={self.tail!r}")
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(line + "\n")


_FOL_SYMBOLS = ("FICO(x), DTI(x), LTV(x), Income(x), LoanAmount(x), YearsAtAddress(x), "
                "and the predicates SelfEmployed(x), ExistingCustomer(x), Approve(x)")


class TwIL:
    def __init__(self, model_path: str | None = None, repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE,
                 n_ctx: int = 0, max_tokens: int = 0, cache_path: Path | None = None,
                 repeat_penalty: float = 1.0, repeat_window: int = 64):
        self.max_tokens = max_tokens
        self.repeat_penalty, self.repeat_window = repeat_penalty, repeat_window
        self._init_llama(model_path, repo, filename, n_ctx)
        self.cache_path = cache_path
        self.progress_path = cache_path.parent / "twil_progress.log" if cache_path else None
        self.cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
        self.calls = 0
        self.stats: list[dict] = []  # one entry per answer used this run (cached or fresh)

    def _init_llama(self, model_path, repo, filename, n_ctx):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise SystemExit("--twil needs llama-cpp-python:  pip install -r requirements-twil.txt") from e
        if model_path is None:
            from huggingface_hub import hf_hub_download
            print(f"  fetching {repo}/{filename} (cached by huggingface_hub after the first run)...")
            model_path = hf_hub_download(repo_id=repo, filename=filename)
        self.model_name = Path(model_path).name
        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=-1,
                         last_n_tokens_size=self.repeat_window, verbose=False)
        self.context = self.llm.n_ctx()
        self.runtime = "llama.cpp"

    def ask(self, prompt: str, job: str = "ask") -> str:
        settings = f"{self.model_name}\n{self.max_tokens}"
        if self.repeat_penalty != 1.0:
            settings += f"\nrp={self.repeat_penalty}/{self.repeat_window}"
        key = hashlib.sha256(f"{settings}\n{prompt}".encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is None:
            t0 = time.time()
            text, tokens, finish = self._gen_llama(prompt, job)
            hit = {"text": text, "tokens": tokens, "seconds": round(time.time() - t0, 2), "finish": finish}
            self.calls += 1
            self.cache[key] = hit
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, indent=1))
        if isinstance(hit, str):  # cache written by an older version of this file
            hit = {"text": hit, "tokens": None, "seconds": None, "finish": None}
        self.stats.append({k: hit[k] for k in ("tokens", "seconds", "finish")})
        return hit["text"]

    def _gen_llama(self, prompt: str, job: str):
        budget = self.max_tokens if self.max_tokens > 0 else self.context
        prog = Progress(self.progress_path, job, budget)
        parts, finish = [], None
        for chunk in self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=self.max_tokens if self.max_tokens > 0 else None,
            repeat_penalty=self.repeat_penalty, stream=True,
        ):
            choice = chunk["choices"][0]
            piece = choice["delta"].get("content")
            if piece:
                parts.append(piece)
                prog.step(piece)  # llama.cpp streams one token per chunk
            finish = choice.get("finish_reason") or finish
        prog.done(finish or "stop")
        return "".join(parts), prog.n, finish or "stop"

    def summary(self) -> dict:
        known = [s for s in self.stats if s["tokens"] is not None]
        tokens = sum(s["tokens"] for s in known)
        seconds = sum(s["seconds"] for s in known)
        return {"answers": len(self.stats), "tokens": tokens, "seconds": round(seconds, 1),
                "tok_per_s": round(tokens / seconds, 1) if seconds else None,
                "max_tokens_one_answer": max((s["tokens"] for s in known), default=0),
                "length_stops": sum(s["finish"] == "length" for s in known)}

    # -- the three jobs -----------------------------------------------------

    def translate(self, clause_text: str) -> str:
        return self.ask(
            "Translate this lending-policy requirement into first-order logic.\n"
            f"Use exactly these function symbols: {_FOL_SYMBOLS}.\n"
            "Write percentages as plain numbers (45% is 45).\n"
            f'Requirement: "{clause_text}"', job="translate",
        )

    def entail(self, premise: str, hypothesis: str) -> tuple[str, str]:
        raw = self.ask(
            f"Premise: {premise}\nHypothesis: {hypothesis}\n"
            "Does the premise entail the hypothesis? Answer entailment, contradiction, or neutral.", job="entail",
        )
        return parse_label(raw), raw

    def induce(self, rows: pd.DataFrame, label: np.ndarray, direction: str) -> str:
        outcome = "declined anyway" if direction == "declined-anyway" else "approved anyway"
        expected = "approved" if direction == "declined-anyway" else "declined"
        lines = [f"{i + 1}. {row_text(r)} -> {int(y)}" for i, (r, y) in enumerate(zip(rows.to_dict("records"), label))]
        return self.ask(
            f"Each line is a loan application from one bank branch that the bank's model expected to be {expected}.\n"
            f"Label 1 means the underwriter {outcome}; label 0 means the expected decision was made.\n"
            "Induce the simplest rule over the attributes that identifies label 1.\n"
            f"Attributes: {', '.join(ORDER)} (self_employed and existing_customer are 0/1).\n"
            "Write the rule as one logical formula using those attribute names, comparison operators, and and/or/not.\n\n"
            + "\n".join(lines) + "\n\nRule:", job="induce",
        )


def row_text(r: dict) -> str:
    parts = []
    for f in ORDER:
        v = r[f]
        parts.append(f"{f}={int(v)}" if FEATURES[f].kind != "real" else f"{f}={v:.1f}")
    return ", ".join(parts)


def parse_label(raw: str) -> str:
    text = logic.strip_think(raw).lower()
    hits = [(m.start(), m.group(0)) for m in re.finditer(r"entailment|contradiction|neutral", text)]
    return hits[-1][1] if hits else "unparsed"


def sample_examples(sub: pd.DataFrame, label: np.ndarray, k: int = 20, seed: int = 0):
    """A balanced, shuffled sample of k positives and k negatives for in-context induction."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(label), np.flatnonzero(~label)
    idx = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False),
                          rng.choice(neg, min(k, len(neg)), replace=False)])
    rng.shuffle(idx)
    return sub.iloc[idx].reset_index(drop=True), label[idx]

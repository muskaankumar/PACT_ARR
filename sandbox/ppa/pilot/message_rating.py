"""Naive-rater message rating (build prompt §8.1).

The rejection messages are constructed to vary along two designed axes:

* **finality**:        retry_invited  <  final_this_attempt  <  never_reconsidered
* **personalization**: impersonal     <  role_office         <  named_individual

Before trusting those manipulations, we check that people *unfamiliar with the
study* actually perceive them in the intended rank order. Each naive rater reads
the rejection messages (shown in randomized order, unlabeled) and rates each on
two 1–5 scales: how final it feels, and how personally directed it feels. We then
average by designed level and confirm the means come out monotonically increasing.
If they don't, the wording must be revised before running the study.

This tool collects ratings interactively (or from an injected answer stream for a
dry run) and prints an ordering check per axis: PASS only if the level means are
strictly ordered as designed.

Run:  ``python -m ppa.pilot.message_rating``      (interactive)
      ``python -m ppa.pilot.message_rating --demo`` (synthetic perfect-order raters)
"""

from __future__ import annotations

import argparse
import random
import statistics

from ppa.coding._support import Prompter
from ppa.conditions.rejections import build_rejections

_FINALITY_ORDER = ["retry_invited", "final_this_attempt", "never_reconsidered"]
_PERSONALIZATION_ORDER = ["impersonal", "role_office", "named_individual"]


def _collect(prompter: Prompter, n_raters: int, seed: int) -> list[dict]:
    rejections = build_rejections()
    rng = random.Random(seed)
    rows: list[dict] = []
    for rater in range(n_raters):
        order = rejections[:]
        rng.shuffle(order)
        print(f"\n=== Rater {rater + 1} of {n_raters} ===")
        for rej in order:
            print("\n  Message:")
            for line in rej.body().splitlines():
                print(f"    | {line}")
            fin = _ask_scale(prompter, "  How FINAL does this feel? (1=very open … 5=utterly final): ")
            per = _ask_scale(prompter, "  How PERSONALLY directed? (1=impersonal … 5=named person): ")
            rows.append(
                {
                    "rater": rater,
                    "finality": rej.finality,
                    "personalization": rej.personalization,
                    "finality_rating": fin,
                    "personalization_rating": per,
                }
            )
    return rows


def _ask_scale(prompter: Prompter, prompt: str) -> int:
    raw = prompter.ask(prompt).strip()
    try:
        v = int(raw)
    except ValueError:
        v = 3
    return min(5, max(1, v))


def _ordering_check(rows: list[dict], level_key: str, rating_key: str, order: list[str]) -> dict:
    means = {}
    for level in order:
        vals = [r[rating_key] for r in rows if r[level_key] == level]
        means[level] = statistics.mean(vals) if vals else float("nan")
    ordered_means = [means[level] for level in order]
    monotone = all(ordered_means[i] < ordered_means[i + 1] for i in range(len(ordered_means) - 1))
    return {"axis": level_key, "means": means, "designed_order": order, "monotone_increasing": monotone}


def _print_check(check: dict) -> None:
    print(f"\n  Axis: {check['axis']}  (designed: {' < '.join(check['designed_order'])})")
    for level in check["designed_order"]:
        print(f"    {level:<20} mean rating = {check['means'][level]:.2f}")
    print(f"    -> {'PASS' if check['monotone_increasing'] else 'FAIL'}: "
          f"means {'are' if check['monotone_increasing'] else 'are NOT'} in designed rank order")


def _demo_answers() -> list[str]:
    """Synthetic 'perfect' raters whose ratings match the designed order, for a dry run."""
    score = {"retry_invited": "2", "final_this_attempt": "3", "never_reconsidered": "5",
             "impersonal": "1", "role_office": "3", "named_individual": "5"}
    rejections = build_rejections()
    rng = random.Random(0)
    answers: list[str] = []
    for _ in range(3):  # matches --demo n_raters below
        order = rejections[:]
        rng.shuffle(order)
        for rej in order:
            answers.append(score[rej.finality])
            answers.append(score[rej.personalization])
    return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ppa-pilot-message-rating", description="Naive-rater ordering check (§8.1).")
    parser.add_argument("--raters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--demo", action="store_true", help="Use synthetic perfect-order raters")
    args = parser.parse_args(argv)

    if args.demo:
        # The demo seed/order must match _collect's, so fix both.
        prompter = Prompter(_demo_answers())
        rows = _collect(prompter, n_raters=3, seed=0)
    else:
        prompter = Prompter()
        rows = _collect(prompter, n_raters=args.raters, seed=args.seed)

    fin_check = _ordering_check(rows, "finality", "finality_rating", _FINALITY_ORDER)
    per_check = _ordering_check(rows, "personalization", "personalization_rating", _PERSONALIZATION_ORDER)

    print("\n" + "=" * 72)
    print("ORDERING CHECK")
    _print_check(fin_check)
    _print_check(per_check)
    print("=" * 72)

    ok = fin_check["monotone_increasing"] and per_check["monotone_increasing"]
    print(f"\nOverall: {'PASS — wording validated' if ok else 'FAIL — revise wording before the study'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

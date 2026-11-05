"""Measure the local model on nemesis matching: accuracy and latency per prompt style.

    uv run python aa893/sim/probe_llm.py

Pairs are deliberately hard: the true match often shares few words (thematic paraphrase),
and decoys share surface words. Prints accuracy, false-positive rate and timing.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_api import LLMClient  # noqa: E402

# (nemesis description, true boss "name. description", decoy boss "name. description")
CASES = [
    ("A giant of bronze plates with a burning furnace in its chest.",
     "The Bronze Warden. A towering sentinel of riveted bronze plates, its chest a furnace.",
     "Mother of Tides. A drowned queen wrapped in kelp, her voice pulling sailors under."),
    ("The one who sings sailors to their graves beneath the waves.",
     "Mother of Tides. A drowned queen wrapped in kelp, her voice pulling sailors beneath black water.",
     "Stormcaller Roc. A colossal bird whose wings crack with lightning over the sea cliffs."),
    ("A ruler long dead who still clings to a broken crown.",
     "The Hollow King. A skeletal monarch on a throne of bones, a cracked crown floating above his skull.",
     "The Gilded Liar. A smiling merchant-prince in golden silks and a jewelled crown."),
    ("A beast that grows stronger every time it is wounded.",
     "Ashen Hydra. A many-headed serpent of smoke; each severed neck sprouts two burning heads.",
     "The Bronze Warden. A towering sentinel of riveted bronze plates that never tires."),
    ("Blind justice that cannot forgive a lie.",
     "The Weeping Judge. A blindfolded giant holding scales that drip blood, sentencing all who lie.",
     "The Gilded Liar. A smiling merchant-prince whose every word is a lie and a trap."),
    ("A creature of the forest that strangles life from the land.",
     "Thornmother. A vast tangle of black briars with a woman's face, roots strangling the earth.",
     "Ashen Hydra. A serpent of smoke and cinders that burns the forest to ash."),
    ("A charming noble whose promises are poison.",
     "The Gilded Liar. A smiling merchant-prince in golden silks whose every word is a trap.",
     "The Weeping Judge. A blindfolded giant who punishes every broken promise."),
    ("Lord of the storm that rules the sky above the cliffs.",
     "Stormcaller Roc. A colossal bird whose wings crack with lightning over the cliffs.",
     "Thornmother. A tangle of briars clinging to the cliffs below a stormy sky."),
]

STYLES = {
    "plain": lambda a, b: f"Two descriptions of a monster.\nA: {a}\nB: {b}\nDo A and B describe the same monster?",
    "riddle": lambda a, b: (f"A riddle describes a monster: \"{a}\"\nA monster appears: \"{b}\"\n"
                            "Is the monster that appears the one the riddle describes?"),
}


def main() -> None:
    llm = LLMClient()
    print("health", llm.health())
    for name, style in STYLES.items():
        tp = fp = 0
        times = []
        for nem, true, decoy in CASES:
            for boss, is_true in ((true, True), (decoy, False)):
                t = time.monotonic()
                ans = llm.yes_no(style(nem, boss), timeout=25, temperature=0)
                times.append(time.monotonic() - t)
                if is_true and ans:
                    tp += 1
                if not is_true and ans:
                    fp += 1
        n = len(CASES)
        print(f"{name:7} TPR {tp}/{n}  FPR {fp}/{n}  time median {statistics.median(times):.2f}s "
              f"max {max(times):.2f}s")
    # choose-among-options style: which of the seen-so-far bosses best fits
    t = time.monotonic()
    ok = 0
    for nem, true, decoy in CASES:
        a = llm.choose(f"Riddle: \"{nem}\"\nWhich monster does the riddle describe?", [true[:120], decoy[:120]],
                       timeout=25, temperature=0)
        ok += a == true[:120]
    print(f"choose  {ok}/{len(CASES)} correct, {(time.monotonic() - t) / len(CASES):.2f}s each")


if __name__ == "__main__":
    main()

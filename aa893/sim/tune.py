"""Random search over PARAMS, scored by mean rank in simulated games.

    uv run python aa893/sim/tune.py --candidates 60 --games 1200 --top 6 --confirm 4000

Stage 1 scores every candidate on the same seeds (common random numbers, so differences are
not luck). Stage 2 re-scores the best few on fresh seeds, plus per-team-archetype checks, so a
setting that only wins against one kind of teammate is visible. Results go to tune_results.json
next to this file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine import ARCHETYPES  # noqa: E402
from run import evaluate  # noqa: E402

SPACE = {
    "prior_attack": (0.6, 0.95),
    "prior_honor": (0.05, 0.6),
    "public_honor_scale": (0.4, 1.0),
    "rally_boost": (0.0, 0.15),
    "ask_cost": (0.0, 0.3),
    "others_vendetta": (0.2, 0.5),
    "gm_attr": (0.7, 0.95),
    "nemesis_threshold": (0.15, 0.6),
    "recip_bonus": (0.0, 0.1),
    "stat_weight": (0.0, 0.02),
    "alive_weight": (0.0, 1.5),
    "vend_weight": (0.0, 1.0),
    "help_margin": (-0.05, 0.3),
    "lie_level": [0, 1, 2],
}


def sample(rng: random.Random) -> dict:
    out = {}
    for k, v in SPACE.items():
        out[k] = rng.choice(v) if isinstance(v, list) else round(rng.uniform(*v), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--confirm", type=int, default=4000)
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "tune_results.json"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = {"rollouts": a.rollouts}
    cands = [{}] + [sample(rng) for _ in range(a.candidates)]     # {} = current defaults
    scored = []
    for i, c in enumerate(cands):
        r = evaluate({**base, **c}, a.games, seed0=10_000)
        scored.append((r["mean_rank"], c, r))
        print(f"[{i + 1}/{len(cands)}] rank {r['mean_rank']:.3f} surv {r['survival']:.3f} "
              f"vend {r['vendetta']:.3f} helps {r['helps']} asks {r['asks']}", flush=True)
    scored.sort(key=lambda x: x[0])
    final = []
    baseline = evaluate(base, a.confirm, seed0=500_000, baseline=True)
    print("baseline", baseline, flush=True)
    for rank1, c, r1 in scored[: a.top]:
        r2 = evaluate({**base, **c}, a.confirm, seed0=500_000)
        per_team = {k: evaluate({**base, **c}, 300, seed0=900_000, team=k)["mean_rank"]
                    for k in list(ARCHETYPES) + ["mixed"]}
        final.append({"params": c, "stage1": r1, "confirm": r2, "per_team": per_team,
                      "worst_team": max(per_team.items(), key=lambda kv: kv[1])})
        print("confirm", r2["mean_rank"], "worst", final[-1]["worst_team"], c, flush=True)
    final.sort(key=lambda f: f["confirm"]["mean_rank"])
    Path(a.out).write_text(json.dumps({"baseline": baseline, "final": final}, indent=1))
    print("best", json.dumps(final[0], indent=1))


if __name__ == "__main__":
    main()

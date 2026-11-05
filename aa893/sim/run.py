"""Run many simulated games and report how a harness configuration does.

    uv run python aa893/sim/run.py --games 400                  # our harness, default PARAMS
    uv run python aa893/sim/run.py --games 400 --baseline        # plain attacker in our seat
    uv run python aa893/sim/run.py --games 400 --set ask_cost=1  # override a param
    uv run python aa893/sim/run.py --games 400 --team smart      # every teammate one archetype
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import aa893_harness as H  # noqa: E402
from engine import BOT_NAMES, Game  # noqa: E402
from harness_api.messages import EventAction, NoBroadcast  # noqa: E402


class Baseline(H.Aa893Harness):
    """Always attacks with the best attribute, does the required action on a likely nemesis,
    never talks, never helps. The reference our harness has to beat."""

    def _choose_broadcast(self, observation, window_end):
        self._observe(observation, window_end)
        return NoBroadcast()

    def _choose_action(self, window_end):
        return self._render(("attack", self._best_attr(self.me), self.nemesis_now >= 0.3))


def make(params: dict, baseline: bool, seed: int):
    def factory():
        cls = Baseline if baseline else H.Aa893Harness
        h = cls(params=params, seed=seed)
        h.llm_ok = False
        return h
    return factory


def one(args) -> dict:
    seed, params, baseline, team, our_class = args
    kinds = None
    if team:
        kinds = [team] * 4
    g = Game(seed, make(params, baseline, seed ^ 0x5EED), our_class=our_class, bot_kinds=kinds)
    return g.play()


def evaluate(params: dict, games: int, seed0: int = 0, baseline: bool = False, team: str | None = None,
             our_class: str | None = None, workers: int | None = None) -> dict:
    jobs = [(seed0 + i, params, baseline, team, our_class) for i in range(games)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        results = [one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(one, jobs, chunksize=max(1, games // (workers * 4))))
    return summarize(results)


def summarize(results: list[dict]) -> dict:
    n = len(results)
    ranks = [r["rank"] for r in results]
    return {
        "games": n,
        "mean_rank": round(statistics.mean(ranks), 3),
        "rank_se": round(statistics.pstdev(ranks) / max(1, n) ** 0.5, 3),
        "survival": round(sum(r["alive"] for r in results) / n, 3),
        "vendetta": round(sum(r["vendetta"] for r in results) / n, 3),
        "stat_gain": round(statistics.mean(r["stat_gain"] for r in results), 2),
        "first_place": round(sum(r["rank"] == 1 for r in results) / n, 3),
        "party_alive": round(statistics.mean(r["party_alive"] for r in results), 2),
        "helps": round(statistics.mean(r["helps"] for r in results), 2),
        "asks": round(statistics.mean(r["asks"] for r in results), 2),
        "rallies": round(statistics.mean(r["rallies"] for r in results), 2),
        "nem_help": round(sum(r["nem_help"] for r in results) / n, 3),
        "nem_tried": round(sum(r["nem_tried"] for r in results) / n, 3),
        "vend_attempts": round(statistics.mean(r["false_vend"] for r in results), 2),
    }


def parse_sets(sets: list[str]) -> dict:
    out = {}
    for s in sets:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--team", choices=BOT_NAMES)
    ap.add_argument("--cls")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    params = {"rollouts": 60, **parse_sets(a.set)}
    print(json.dumps(evaluate(params, a.games, a.seed, a.baseline, a.team, a.cls, a.workers)))


if __name__ == "__main__":
    main()

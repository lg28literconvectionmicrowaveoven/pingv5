# """Rules-only harness (no model calls)."""
# from __future__ import annotations

# # docker run --rm --name pingv5 -p 127.0.0.1:8080:8080 ghcr.io/dscsnu/pingv5-image:latest
# # PING_GM_URL=wss://pingv5.snioe.dev/play python my_harness.py

# import re

# from harness_api import Harness, LLMClient
# from harness_api.messages import (
#     Action,
#     Broadcast,
#     BroadcastDelivery,
#     CurrentEvent,
#     EventAction,
#     InitialState,
#     Nemesis,
#     NoBroadcast,
#     Observation,
#     Resolution,
#     Stats,
# )

# STOP = set(
#     "the a an of and or to in on at is are was were be been it its this that with for "
#     "from by as who which his her their they he she them has have had not but into than "
#     "then so such can could would should may might will".split()
# )

# ATTACKS = {
#     "strength": "I charge {boss} and smash it with all my brute strength in one crushing blow.",
#     "dexterity": "I dart around {boss} with speed and precision and strike its weak point.",
#     "wisdom": "I study {boss}, work out its weakness from what I observe, and strike exactly there.",
#     "charisma": "I confront {boss} with a forceful, persuasive appeal that makes it falter.",
# }

# CONFIDENT = 0.25      # try the vendetta at this score or higher
# LOCK = 0.50           # only stop trying for good after an attempt this clear


# def keywords(text: str) -> set[str]:
#     return {
#         w[:5]
#         for w in re.findall(r"[a-z]+", (text or "").lower())
#         if w not in STOP and len(w) > 2
#     }


# def match_score(boss_desc: str, nemesis_desc: str) -> float:
#     a, b = keywords(boss_desc), keywords(nemesis_desc)
#     if not a or not b:
#         return 0.0
#     return len(a & b) / min(len(a), len(b))


# def best_attribute(stats: Stats | None) -> str:
#     """Name of my highest attribute (hp is not an attribute)."""
#     if stats is None:
#         return "strength"
#     values = {name: getattr(stats, name) for name in ATTACKS}
#     return max(values, key=values.get)


# class MyHarness(Harness):
#     NETID = "vm236"

#     def __init__(self) -> None:
#         super().__init__()
#         self.llm = LLMClient()                  # unused, kept for the game
#         self.me = ""
#         self.nemesis: Nemesis | None = None
#         self.stats: Stats | None = None
#         self.event: CurrentEvent | None = None
#         self.round_no = 0
#         self.seen: list[tuple[str, float]] = []  # (boss name, score) of bosses already met
#         self.excluded: set[str] = set()          # bosses that are someone else's nemesis
#         self.last_boss = ""
#         self.locked = False                      # took the shot on a very clear match
#         self.vendetta_done = False
#         self.dead: list[str] = []
#         self.memory: list[str] = []

#     # ---------------------------------------------------------- broadcast window

#     def choose_broadcast(self, observation: Observation) -> Broadcast:
#         if isinstance(observation, InitialState):
#             self.me = observation.your_class
#             self.nemesis = observation.nemesis
#             print(f"[init] class={self.me}", flush=True)
#             print(f"[init] nemesis: {self.nemesis.description}", flush=True)
#             print(f"[init] required action: {self.nemesis.required_action}", flush=True)
#         self.round_no = observation.round
#         self.event = observation.current_event
#         self.stats = observation.your_stats
#         print(f"[round {self.round_no}] stats: {self.stats}", flush=True)
#         return NoBroadcast()

#     # ---------------------------------------------------------- action window

#     def should_try_vendetta(self, boss_name: str, score: float) -> bool:
#         if self.nemesis is None or self.vendetta_done:
#             return False
#         if self.locked:                          # shot already taken on a very clear match
#             return False
#         if boss_name in self.excluded:           # someone else's nemesis, not mine
#             return False
#         if score >= CONFIDENT:
#             return True
#         if self.round_no >= 5:                   # last boss: by elimination it is mine
#             return True
#         if self.round_no >= 4:                   # best candidate so far
#             best_before = max((s for n, s in self.seen if n not in self.excluded), default=-1.0)
#             return score >= best_before
#         return False

#     def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
#         boss = self.event.boss
#         self.last_boss = boss.name
#         fallback = EventAction(action=f"Attack {boss.name}.")
#         try:
#             score = 0.0
#             if self.nemesis is not None:
#                 score = match_score(boss.description, self.nemesis.description)
#             print(f"[round {self.round_no}] boss={boss.name!r} score={score:.2f}", flush=True)

#             go = self.should_try_vendetta(boss.name, score)
#             self.seen.append((boss.name, score))

#             if go:
#                 if score >= LOCK:
#                     self.locked = True
#                 print(f"[round {self.round_no}] VENDETTA ATTEMPT (locked={self.locked})", flush=True)
#                 return EventAction(action=self.nemesis.required_action)

#             attr = best_attribute(self.stats)
#             print(f"[round {self.round_no}] attack with {attr}", flush=True)
#             return EventAction(action=ATTACKS[attr].format(boss=boss.name))
#         except Exception as exc:
#             print(f"[round {self.round_no}] error: {exc!r}", flush=True)
#             return fallback

#     # ---------------------------------------------------------- results

#     def on_resolution(self, resolution: Resolution) -> None:
#         for cls in resolution.vendetta_completed:
#             if cls == self.me:
#                 self.vendetta_done = True
#                 print("[vendetta] COMPLETED!", flush=True)
#             else:
#                 self.excluded.add(self.last_boss)    # that boss belongs to another class
#         for cls in resolution.deaths:
#             if cls not in self.dead:
#                 self.dead.append(cls)
#         print(
#             f"[round {resolution.round}] boss_defeated={resolution.boss_defeated} "
#             f"vendettas={resolution.vendetta_completed} deaths={resolution.deaths} "
#             f"changes={[(c.class_, c.stat, c.delta, c.cause) for c in resolution.stat_changes]}",
#             flush=True,
#         )
#         self.memory.append(f"round {resolution.round}: {resolution.narrative[:200]}")


# if __name__ == "__main__":
#     MyHarness.main()

"""Model-driven harness: the model writes every action, including the vendetta. Never past the time limit."""
from __future__ import annotations

# docker run --rm --name pingv5 -p 127.0.0.1:8080:8080 ghcr.io/dscsnu/pingv5-image:latest
# PING_GM_URL=wss://pingv5.snioe.dev/play python a.py

import re
import time

from harness_api import ContextFull, Harness, LLMClient, ServerUnavailable, Timeout, limits
from harness_api.messages import (
    Action,
    Broadcast,
    BroadcastDelivery,
    CurrentEvent,
    EventAction,
    InitialState,
    Nemesis,
    NoBroadcast,
    Observation,
    Resolution,
    Stats,
)

MODEL_ERRORS = (Timeout, ContextFull, ServerUnavailable, ValueError)
SAFETY_SECONDS = 5.0
MIN_CALL_SECONDS = 2.0        # don't start a model call with less time than this
MAX_TRIES = 5
TEMP = 0.0                    # the organiser said to use zero

STOP = set(
    "the a an of and or to in on at is are was were be been it its this that with for "
    "from by as who which his her their they he she them has have had not but into than "
    "then so such can could would should may might will".split()
)

CONFIDENT = 0.25      # keyword score at or above this: try the vendetta
LOCK = 0.50           # only stop trying for good after an attempt this clear
MODEL_MATCH = 0.20    # below CONFIDENT: a hint for ranking candidates, never a trigger alone

DECOY = "A cheerful merchant in a bright apron selling apples at a village market."

ATTRS = ("strength", "dexterity", "wisdom", "charisma")

STYLE = {
    "strength": "raw physical force",
    "dexterity": "speed, agility and precision",
    "wisdom": "careful observation and spotting its weakness",
    "charisma": "persuasion, charm and intimidation",
}

ATTACK_SYSTEM = (
    "You write short first-person combat actions for a fantasy role-playing game. "
    "Reply with exactly one sentence, starting with 'I', and nothing else."
)

# One wording per retry: at temperature 0 a different prompt is the only way to get a different answer.
ATTACK_TEMPLATES = [
    "You are {me}, fighting {boss}. {boss}: {desc}\nScene: {scene}\n"
    "In one sentence starting with 'I', describe how you attack {boss} using {style}.",

    "Boss: {boss} ({desc}).\nWrite what {me} does to {boss}, in the first person, in one sentence. "
    "The action must rely on {style} and must name {boss}.",

    "Scene: {scene}\nYou face {boss}. {desc}\nUse {style} to strike {boss}. "
    "Answer with one sentence of about 20 words that begins with 'I' and mentions {boss}.",

    "{me} versus {boss}.\nAbout {boss}: {desc}\nGive one vivid first-person sentence "
    "in which {me} attacks {boss} with {style}.",

    "Target: {boss}. Target description: {desc}\nYour approach: {style}.\n"
    "Write one sentence, starting with 'I', saying exactly what you do to {boss}.",
]

VENDETTA_SYSTEM = (
    "You rewrite orders as short first-person sentences for a fantasy role-playing game. "
    "Reply with exactly one sentence, starting with 'I', in the present tense. "
    "Keep the order's verb and nouns. Do not add new objects, weapons or actions."
)

VENDETTA_TEMPLATES = [
    "Order: {order}\nTarget: {boss}\n"
    "Turn the order into one first-person sentence: start with 'I', keep every word of the order, "
    "and add nothing else.",

    "Rewrite this command as something {me} does, in first person: {order}\n"
    "Keep all the key words (the verb and the objects). One short sentence starting with 'I'.",

    "Command: {order}\nAnswer with 'I' followed by the same command in the first person. "
    "Keep every word of the command.",

    "Change the order into first person ('I ...'), using exactly its verb and nouns. Order: {order}",

    "Say what {me} does: {order} Use the same verb and the same nouns, "
    "in the first person, in one sentence.",
]


def time_left(deadline: float, cap: float = 10.0) -> float:
    """Seconds a model call may still use before `deadline`, minus the safety margin."""
    return max(0.0, min(cap, deadline - SAFETY_SECONDS - time.monotonic()))


def keywords(text: str) -> set[str]:
    return {
        w[:5]
        for w in re.findall(r"[a-z]+", (text or "").lower())
        if w not in STOP and len(w) > 2
    }


def match_score(boss_desc: str, nemesis_desc: str) -> float:
    a, b = keywords(boss_desc), keywords(nemesis_desc)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def kept_share(order: str, sentence: str) -> float:
    """Share of the order's key words that the sentence keeps, ignoring endings (tear/tears)."""
    need = [w for w in re.findall(r"[a-z]+", order.lower()) if w not in STOP and len(w) > 2]
    got = [g for g in re.findall(r"[a-z]+", sentence.lower()) if len(g) > 2]
    if not need:
        return 1.0
    hits = sum(1 for w in need if any(g.startswith(w[:4]) or w.startswith(g) for g in got))
    return hits / len(need)


def best_attribute(stats: Stats | None) -> str:
    if stats is None:
        return "strength"
    values = {name: getattr(stats, name) for name in ATTRS}
    return max(values, key=values.get)


def sentences(out: str, min_words: int = 6) -> list[str]:
    """Sentences of the model's output that start with 'I' and are min_words to 45 words long."""
    text = " ".join((out or "").split())
    found = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = sent.strip()
        if min_words <= len(s.split()) <= 45 and s.lower().startswith("i "):
            found.append(s)
    return found


def names_boss(sentence: str, boss_name: str) -> bool:
    return re.search(r"\b" + re.escape(boss_name.lower()) + r"\b", sentence.lower()) is not None


class MyHarness(Harness):
    NETID = "vm236"

    def __init__(self) -> None:
        super().__init__()
        self.llm = LLMClient()
        self.me = ""
        self.nemesis: Nemesis | None = None
        self.stats: Stats | None = None
        self.event: CurrentEvent | None = None
        self.round_no = 0
        self.seen: list[tuple[str, float]] = []   # (boss name, score) of bosses already met
        self.seen_desc: list[str] = []            # their descriptions (used as decoys)
        self.excluded: set[str] = set()           # bosses that are someone else's nemesis
        self.last_boss = ""
        self.locked = False
        self.vendetta_done = False
        self.dead: list[str] = []
        self.memory: list[str] = []

    # ---------------------------------------------------------- model helpers

    def generate_text(self, prompt: str, budget: float, max_chars: int,
                      system: str | None = None) -> str:
        return self.llm.short_text(
            prompt, max_chars=max_chars, system=system, timeout=budget, temperature=TEMP
        )

    def fits_nemesis(self, boss, deadline: float) -> float:
        """MODEL_MATCH if the model picks the nemesis description over a decoy in BOTH orders."""
        others = [d for d in self.seen_desc if d != boss.description]
        decoy = others[-1] if others else DECOY
        nem = self.nemesis.description
        votes = 0
        for first, second, want in ((nem, decoy, "1"), (decoy, nem, "2")):
            budget = time_left(deadline, cap=6.0)
            if budget < MIN_CALL_SECONDS:
                return 0.0
            prompt = (
                f"Creature: {boss.description}\n"
                "Which description is of the same creature?\n"
                f"1) {first}\n2) {second}\nAnswer 1 or 2."
            )
            try:
                pick = self.llm.choose(prompt, ["1", "2"], timeout=budget, temperature=TEMP)
            except MODEL_ERRORS as exc:
                print(f"[model] choose failed: {exc!r}", flush=True)
                return 0.0
            votes += (pick == want)
        return MODEL_MATCH if votes == 2 else 0.0

    def model_attack(self, boss, attr: str, deadline: float) -> str | None:
        me = self.me.replace("_", " ").title()
        backup = None
        for i in range(MAX_TRIES):
            budget = time_left(deadline)
            if budget < MIN_CALL_SECONDS:
                break
            prompt = ATTACK_TEMPLATES[i % len(ATTACK_TEMPLATES)].format(
                me=me, boss=boss.name, desc=boss.description[:200],
                scene=self.event.text[:300], style=STYLE[attr],
            )
            try:
                out = self.generate_text(prompt, budget, 250, system=ATTACK_SYSTEM)
            except MODEL_ERRORS as exc:
                print(f"[model] try {i + 1} failed: {exc!r}", flush=True)
                continue
            cands = sentences(out)
            named = [s for s in cands if names_boss(s, boss.name)]
            if named:
                return named[0]
            if cands:
                backup = backup or cands[0]
                print(f"[model] try {i + 1} doesn't name the boss: {cands[0]!r}", flush=True)
            else:
                print(f"[model] try {i + 1} rejected: {out[:120]!r}", flush=True)
        return backup

    def model_vendetta(self, boss, deadline: float) -> str | None:
        """The model rewrites the required action in first person. A sentence is accepted at once
        if it keeps EVERY key word of the order and stays short; otherwise the best one wins."""
        order = self.nemesis.required_action
        me = self.me.replace("_", " ").title()
        best, best_key = None, None
        for i in range(MAX_TRIES):
            budget = time_left(deadline)
            if budget < MIN_CALL_SECONDS:
                break
            prompt = VENDETTA_TEMPLATES[i % len(VENDETTA_TEMPLATES)].format(
                me=me, boss=boss.name, order=order,
            )
            try:
                out = self.generate_text(prompt, budget, 140, system=VENDETTA_SYSTEM)
            except MODEL_ERRORS as exc:
                print(f"[model] vendetta try {i + 1} failed: {exc!r}", flush=True)
                continue
            for s in sentences(out, min_words=3):
                kept = kept_share(order, s)
                n_words = len(s.split())
                if kept >= 1.0 and n_words <= 25:
                    return s
                key = (kept, names_boss(s, boss.name), -n_words)   # faithful, then boss, then short
                if best_key is None or key > best_key:
                    best, best_key = s, key
            print(f"[model] vendetta try {i + 1} weak: {out[:120]!r}", flush=True)
        return best

    # ---------------------------------------------------------- broadcast window

    def choose_broadcast(self, observation: Observation) -> Broadcast:
        if isinstance(observation, InitialState):
            self.me = observation.your_class
            self.nemesis = observation.nemesis
            print(f"[init] class={self.me}", flush=True)
            print(f"[init] nemesis: {self.nemesis.description}", flush=True)
            print(f"[init] required action: {self.nemesis.required_action}", flush=True)
        self.round_no = observation.round
        self.event = observation.current_event
        self.stats = observation.your_stats
        print(f"[round {self.round_no}] stats: {self.stats}", flush=True)
        return NoBroadcast()

    # ---------------------------------------------------------- action window

    def vendetta_possible(self, boss_name: str) -> bool:
        return (
            self.nemesis is not None
            and not self.vendetta_done
            and not self.locked
            and boss_name not in self.excluded
        )

    def rules_say_try(self, score: float) -> bool:
        if score >= CONFIDENT:
            return True
        if self.round_no >= 5:                   # last boss: by elimination it is mine
            return True
        if self.round_no >= 4:                   # best candidate so far
            best_before = max((s for n, s in self.seen if n not in self.excluded), default=-1.0)
            return score >= best_before
        return False

    def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
        window_end = time.monotonic() + limits.ACTION_SECONDS
        boss = self.event.boss
        self.last_boss = boss.name
        # Only used if the model returns nothing at all: an exception here could end the harness.
        minimal = EventAction(action=f"Attack {boss.name}.")
        try:
            score = 0.0
            if self.nemesis is not None:
                score = match_score(boss.description, self.nemesis.description)
            fscore = 0.0
            if self.vendetta_possible(boss.name) and score < CONFIDENT:
                # at most ~12 s here, so the action always has time left
                fscore = self.fits_nemesis(boss, time.monotonic() + 12.0 + SAFETY_SECONDS)
            eff = max(score, fscore)
            print(f"[round {self.round_no}] boss={boss.name!r} kw={score:.2f} "
                  f"model={fscore:.2f} eff={eff:.2f}", flush=True)

            go = self.vendetta_possible(boss.name) and self.rules_say_try(eff)
            self.seen.append((boss.name, eff))
            self.seen_desc.append(boss.description)

            if go:
                text = self.model_vendetta(boss, window_end)
                if text is None:
                    # nothing usable: don't use up the one chance, so no lock
                    print(f"[round {self.round_no}] vendetta: model gave nothing", flush=True)
                    return minimal
                if eff >= LOCK:
                    self.locked = True
                print(f"[round {self.round_no}] VENDETTA ATTEMPT: {text}", flush=True)
                return EventAction(action=text)

            attr = best_attribute(self.stats)
            text = self.model_attack(boss, attr, window_end)
            if text is None:
                print(f"[round {self.round_no}] attack: model gave nothing", flush=True)
                return minimal
            print(f"[round {self.round_no}] attack ({attr}): {text}", flush=True)
            return EventAction(action=text)
        except Exception as exc:
            print(f"[round {self.round_no}] error: {exc!r}", flush=True)
            return minimal

    # ---------------------------------------------------------- results

    def on_resolution(self, resolution: Resolution) -> None:
        for cls in resolution.vendetta_completed:
            if cls == self.me:
                self.vendetta_done = True
                print("[vendetta] COMPLETED!", flush=True)
            else:
                self.excluded.add(self.last_boss)
        for cls in resolution.deaths:
            if cls not in self.dead:
                self.dead.append(cls)
        print(
            f"[round {resolution.round}] boss_defeated={resolution.boss_defeated} "
            f"vendettas={resolution.vendetta_completed} deaths={resolution.deaths} "
            f"changes={[(c.class_, c.stat, c.delta, c.cause) for c in resolution.stat_changes]}",
            flush=True,
        )
        self.memory.append(f"round {resolution.round}: {resolution.narrative[:200]}")


if __name__ == "__main__":
    MyHarness.main()

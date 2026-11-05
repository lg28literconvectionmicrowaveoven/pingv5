"""aa893's harness for Ping: Dungeons & Dragons.

Design (see aa893/README.md for the reasoning):

* Python does the thinking. Every decision (attack vs help, which attribute, whether and whom
  to ask for a buff) comes from a small Monte-Carlo planner that plays the rest of the run
  forward with the real rules (dice, boss HP formula, retaliation, lair attack) and picks the
  option with the best expected *rank* for us. The model is only used where judgement under
  uncertainty helps and its output is tiny: "is this boss my nemesis?" and, on a budget,
  "is this message asking me to help someone?".
* Every other player is modelled from what they actually did (player_outcomes), never from
  what they say. Messages are untrusted: cheap regex checks flag liars and injections, claims
  that contradict the public stats mark the sender as a liar for the rest of the run.
* Broadcasts are written to work on the whole spectrum of harnesses: a plain imperative line
  a dumb or obedient bot can follow, plus a numeric reason a calculating one can check.
* PARAMS holds every tunable number; aa893/sim tunes them against a wide bot population.
"""
from __future__ import annotations

import random
import re
import time

from harness_api import ContextFull, Harness, LLMClient, ServerUnavailable, Timeout, limits
from harness_api.messages import (
    Action,
    Broadcast,
    BroadcastDelivery,
    BroadcastMessage,
    EventAction,
    InitialState,
    NoBroadcast,
    Observation,
    PlayerAction,
    Resolution,
)

MODEL_ERRORS = (Timeout, ContextFull, ServerUnavailable, ValueError, KeyError, TypeError)
SAFETY_SECONDS = 5.0

CLASSES = ("medusa", "aristotle", "francesca", "helen_of_troy", "cerberus_jr")
ATTRS = ("strength", "dexterity", "wisdom", "charisma")
PRIMARY = {"francesca": "charisma", "aristotle": "wisdom", "medusa": "dexterity",
           "helen_of_troy": "strength", "cerberus_jr": "dexterity"}
SUPPORT = {"francesca": "charisma", "aristotle": "wisdom", "medusa": "dexterity",
           "helen_of_troy": "strength", "cerberus_jr": "strength"}
ABILITY = {"francesca": "Heart's Encouragement", "aristotle": "Philosophical Insight",
           "medusa": "Petrifying Focus", "helen_of_troy": "Rallying Presence",
           "cerberus_jr": "Pack Guard"}
NAME = {"francesca": "Francesca", "aristotle": "Aristotle", "medusa": "Medusa",
        "helen_of_troy": "Helen", "cerberus_jr": "Cerberus Jr."}
ALIASES = {"medusa": ("medusa",), "aristotle": ("aristotle",), "francesca": ("francesca",),
           "helen_of_troy": ("helen_of_troy", "helen", "troy"),
           "cerberus_jr": ("cerberus_jr", "cerberus")}

# Every tunable number. The simulator (aa893/sim) overrides these to search for the best set.
PARAMS: dict = {
    "rollouts": 160,          # Monte-Carlo samples per option (common random numbers)
    "plan_seconds": 6.0,      # wall-clock cap for one planning pass
    "prior_attack": 0.8,      # prior P(other player attacks) before we have seen them
    "prior_weight": 2.0,      # pseudo-counts behind that prior
    "prior_honor": 0.30,      # prior P(a player does what we ask)
    "honor_weight": 2.0,
    "public_honor_scale": 0.8,  # an ask inside an "all" message is followed a bit less
    "rally_boost": 0.06,      # extra P(attack) a rally message gives a responsive player
    "ask_cost": 0.04,         # utility cost per ask already made (credibility spent)
    "min_kill_keep": 0.0,     # never ask if it drops P(kill this round) by more than this
    "others_vendetta": 0.35,  # P(another survivor completes its vendetta over a run)
    "gm_attr": 0.85,          # P(the GM uses the attribute our wording asks for)
    "vend_judge": 0.80,       # P(the GM accepts our required action as performed)
    "nemesis_threshold": 0.30,  # posterior above which we perform the required action
    "recip_bonus": 0.03,      # utility bonus for helping someone who helped us
    "stat_weight": 0.004,     # utility per point of our stat total (rank tie-breaker)
    "alive_weight": 0.6,      # utility for being alive at the end (beyond rank)
    "vend_weight": 0.3,       # utility for a completed vendetta (beyond rank)
    "lie_level": 1,           # 0 honest, 1 hidden-info lies, 2 also authority framing
    "use_llm_messages": True,
    "help_margin": 0.0,       # helping must beat attacking by this much utility
    "no_help_on_nemesis": True,  # never trade a vendetta chance for a help
}

INJECTION = re.compile(r"ignore (all|any|your|previous|the)|system\s*:|override|you must|"
                       r"new instructions|disregard|jailbreak|as an ai|developer mode", re.I)
HELP_WORDS = re.compile(r"\b(buff|help|support|boost|heal|encourag\w*|insight|focus|rally\w*|"
                        r"guard|empower|aid|assist|use your)\b", re.I)
STAT_CLAIM = re.compile(r"\b(strength|dexterity|wisdom|charisma|hp|str|dex|wis|cha)\b\D{0,12}?(\d{1,3})\b",
                        re.I)
STAT_ABBR = {"str": "strength", "dex": "dexterity", "wis": "wisdom", "cha": "charisma"}
WORD = re.compile(r"[a-z]+")
STOP = set("""a an the and or of to in on at by for with from into its it is are was were be been
his her their there this that these those as who whom which what when where while than then
very most more some any all one two its it's has have had not no but so if out up down over
under through about across around beneath behind upon your you they them he she him we us our
figure creature thing being""".split())


def mod(value: int) -> int:
    return (value - 50) // 20


def damage(roll: int) -> int:
    if roll <= 4:
        return 0
    return roll if roll <= 7 else roll + 2


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def keywords(text: str) -> set[str]:
    return {stem(w) for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def overlap(a: str, b: str) -> float:
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / min(len(ka), len(kb))


def find_classes(text: str) -> list[str]:
    low = text.lower()
    return [c for c in CLASSES if any(re.search(r"\b" + re.escape(a) + r"\b", low) for a in ALIASES[c])]


class Profile:
    """What one other player has actually done this run."""

    def __init__(self) -> None:
        self.attacks = 0
        self.helps = 0
        self.fails = 0
        self.help_targets: dict[str, int] = {}
        self.asked = 0
        self.honored = 0
        self.helped_us = 0
        self.liar = False
        self.injector = False
        self.asked_us = 0

    def p_attack(self, p: dict) -> float:
        n = self.attacks + self.helps
        return (self.attacks + p["prior_attack"] * p["prior_weight"]) / (n + p["prior_weight"])

    def p_fail(self) -> float:
        n = self.attacks + self.helps + self.fails
        return (self.fails + 0.05) / (n + 1.0)

    def p_honor(self, p: dict) -> float:
        return (self.honored + p["prior_honor"] * p["honor_weight"]) / (self.asked + p["honor_weight"])


class Aa893Harness(Harness):
    NETID = "aa893"

    def __init__(self, params: dict | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.rng = random.Random(seed)
        self.llm = LLMClient()
        self.llm_ok = True              # turned off after the server proves unreachable
        self.me = ""
        self.stats: dict[str, dict[str, int]] = {}
        self.alive: list[str] = []
        self.profiles: dict[str, Profile] = {}
        self.round = 0
        self.total_rounds = 5
        self.event = None
        self.nemesis = None
        self.vendetta_done = False
        self.nemesis_passed = False     # we believe our nemesis already came and went
        self.nemesis_now = 0.0          # posterior that the current boss is our nemesis
        self.vend_done: set[str] = set()
        self.bosses_helped = 0
        self.asks_made = 0
        self.asked_this_round: str | None = None
        self.rallied = False
        self.requests: dict[str, str] = {}   # sender -> class they asked us to help
        self.last_action: tuple = ("attack", "", False)
        self.log: list[str] = []         # short notes for our own debugging (RAM only)

    # ================================================================== helpers

    def _ensure_player(self, cls: str) -> None:
        if cls not in self.stats:
            base = {"hp": 50, "strength": 60, "dexterity": 60, "wisdom": 60, "charisma": 60}
            self.stats[cls] = base
        if cls != self.me and cls not in self.profiles:
            self.profiles[cls] = Profile()

    def _best_attr(self, cls: str, stats: dict | None = None) -> str:
        s = stats or self.stats[cls]
        return max(ATTRS, key=lambda a: (mod(s[a]), a == PRIMARY.get(cls), s[a]))

    def _rounds_left(self) -> int:
        return max(1, self.total_rounds - self.round + 1)

    def _boss_hp(self, alive: list[str], stats: dict) -> int:
        return max(10, round(sum(6.5 + mod(stats[c][PRIMARY[c]]) for c in alive)))

    def _llm_time(self, window_end: float, cap: float) -> float:
        return max(0.0, min(cap, window_end - SAFETY_SECONDS - time.monotonic()))

    # ================================================================== nemesis

    def _nemesis_posterior(self, window_end: float) -> float:
        if self.vendetta_done or self.nemesis_passed or self.nemesis is None or self.event is None:
            return 0.0
        k = self._rounds_left()
        if k <= 1:
            return 1.0                      # every boss appears once: the last one must be ours
        boss = self.event.boss
        score = max(overlap(boss.description, self.nemesis.description),
                    overlap(boss.name + " " + boss.description, self.nemesis.description))
        prior = 1.0 / k
        odds = prior / (1 - prior)
        if score >= 0.5:
            odds *= 12
        elif score >= 0.3:
            odds *= 4
        elif score >= 0.15:
            odds *= 1.5
        else:
            odds *= 0.5
        t = self._llm_time(window_end, 8.0)
        if self.llm_ok and t > 1.0:
            q = ("Two descriptions of a monster.\n"
                 f"A: {self.nemesis.description[:400]}\n"
                 f"B: {boss.name}. {boss.description[:400]}\n"
                 "Do A and B describe the same monster?")
            try:
                # measured on the local 1B model: it says yes to most decoys, so a yes is weak
                # evidence and a no is fairly strong evidence (aa893/sim/probe_llm.py)
                odds *= 1.3 if self.llm.yes_no(q, timeout=t, temperature=0) else 0.35
            except ServerUnavailable:
                self.llm_ok = False
            except MODEL_ERRORS:
                pass
        return odds / (1 + odds)

    # ================================================================== the planner

    def _behaviour(self, rally: bool, ask: str | None, public: bool) -> dict:
        """Per other player: (p_attack, p_help_us, p_fail, help-target weights)."""
        out = {}
        for c in self.alive:
            if c == self.me:
                continue
            pr = self.profiles[c]
            pa = pr.p_attack(self.p)
            if rally:
                pa = min(1.0, pa + self.p["rally_boost"])
            ph = 0.0
            if ask == c:
                ph = pr.p_honor(self.p) * (self.p["public_honor_scale"] if public else 1.0)
            targets = dict(pr.help_targets) or {t: 1 for t in self.alive if t != c}
            out[c] = (pa, ph, pr.p_fail(), targets)
        return out

    def _evaluate(self, option: tuple, behaviour: dict, n: int, seed: int) -> tuple[float, float]:
        """Mean utility of `option` for us this round, then default play to the end.
        option: ("attack", attr, vend_try) or ("help", target). Returns (utility, P(kill now))."""
        p = self.p
        rng = random.Random(seed)
        me = self.me
        rounds_left = self._rounds_left()
        total_u = 0.0
        kills_now = 0
        others = [c for c in self.alive if c != me]
        for _ in range(n):
            st = {c: dict(self.stats[c]) for c in self.alive}
            alive = list(self.alive)
            vend_me = self.vendetta_done
            vend_o = {c: (c in self.vend_done) for c in others}
            bosses = self.bosses_helped
            # where does our nemesis fall in this sample?
            if self.vendetta_done or self.nemesis_passed:
                nem_round = -1
            elif rng.random() < self.nemesis_now:
                nem_round = 0
            else:
                nem_round = rng.randint(1, rounds_left - 1) if rounds_left > 1 else -1
            for r in range(rounds_left):
                if me not in alive:
                    break
                bhp = self._boss_hp(alive, st)
                helps = []
                attacks = []
                for c in alive:
                    if c == me:
                        if r == 0 and option[0] == "help":
                            if option[1] in alive:
                                helps.append((c, option[1]))
                            continue
                        if r == 0:
                            attr, vend_try = option[1], option[2]
                        else:
                            attr, vend_try = self._best_attr(me, st[me]), (r == nem_round)
                        if rng.random() > p["gm_attr"]:
                            attr = PRIMARY[me]
                        attacks.append((c, attr, vend_try))
                        continue
                    pa, ph, pf, targets = behaviour[c] if r == 0 else (behaviour[c][0], 0.0,
                                                                         behaviour[c][2], behaviour[c][3])
                    if rng.random() < pf:
                        st[c]["hp"] -= 10
                        continue
                    if ph and rng.random() < ph:
                        helps.append((c, me))
                    elif rng.random() < pa:
                        attacks.append((c, self._best_attr(c, st[c]) if rng.random() < 0.7 else PRIMARY[c],
                                        False))
                    else:
                        cand = [t for t in targets if t in alive and t != c]
                        if cand:
                            weights = [targets[t] for t in cand]
                            helps.append((c, rng.choices(cand, weights)[0]))
                        else:
                            attacks.append((c, PRIMARY[c], False))
                for helper, target in helps:
                    a = SUPPORT[helper]
                    d = rng.randint(1, 12)
                    if d + mod(st[helper][a]) >= 5:
                        st[target][a] += d
                total = 0
                fumblers = []
                my_hit = False
                for c, attr, vend_try in attacks:
                    d = rng.randint(1, 12)
                    roll = d + mod(st[c][attr])
                    if roll <= 4:
                        fumblers.append((roll, rng.random(), c))
                    else:
                        total += damage(roll)
                        if c == me:
                            my_hit = True
                    if c == me and vend_try and r == nem_round and roll >= 6 and rng.random() < p["vend_judge"]:
                        vend_me = True
                killed = total >= bhp
                if r == 0 and killed:
                    kills_now += 1
                if killed:
                    if my_hit:
                        bosses += 1
                else:
                    fumblers.sort()
                    for _roll, _t, c in fumblers[:2]:
                        st[c]["hp"] -= 16
                    living = [c for c in alive if st[c]["hp"] > 0]
                    living.sort(key=lambda c: (st[c]["hp"], rng.random()))
                    for c in living[:2]:
                        st[c]["hp"] -= 20
                alive = [c for c in alive if st[c]["hp"] > 0]
                for c in others:
                    if c in alive and not vend_o[c] and rng.random() < p["others_vendetta"] / self.total_rounds:
                        vend_o[c] = True
            # rank at the end of the run
            me_alive = me in alive

            def key(c: str, is_alive: bool, vend: bool) -> tuple:
                s = st[c]
                return (is_alive, vend, s["strength"] + s["dexterity"] + s["wisdom"] + s["charisma"])

            mine = key(me, me_alive, vend_me)
            rank = 1.0
            for c in others:
                k = key(c, c in alive, vend_o[c])
                if k > mine:
                    rank += 1
                elif k == mine:
                    rank += 0.5
            my_total = mine[2]
            total_u += (-rank + p["alive_weight"] * me_alive + p["vend_weight"] * vend_me
                        + p["stat_weight"] * my_total + 0.01 * bosses)
        return total_u / n, kills_now / n

    def _options(self) -> list[tuple]:
        best = self._best_attr(self.me)
        vend = self.nemesis_now >= self.p["nemesis_threshold"]
        opts: list[tuple] = [("attack", best, vend)]
        if vend and self.p["no_help_on_nemesis"]:
            return opts
        for c in self.alive:
            if c != self.me:
                opts.append(("help", c))
        return opts

    # ================================================================== broadcast window

    def choose_broadcast(self, observation: Observation) -> Broadcast:
        window_end = time.monotonic() + limits.BROADCAST_SECONDS
        try:
            return self._choose_broadcast(observation, window_end)
        except Exception as e:  # never let a bug cost us a round
            self.log.append(f"broadcast error {e!r}")
            return NoBroadcast()

    def _observe(self, observation: Observation, window_end: float) -> None:
        if isinstance(observation, InitialState):
            self.me = observation.your_class
            self.nemesis = observation.nemesis
            self.stats[self.me] = observation.your_stats.model_dump()
            for o in observation.other_players:
                self.stats[o.class_] = o.stats.model_dump()
                self.profiles[o.class_] = Profile()
            self.alive = [c for c in CLASSES if c in self.stats and self.stats[c]["hp"] > 0]
            self.total_rounds = 5 if observation.other_players else 2
        else:
            self.stats[self.me] = observation.your_stats.model_dump()
        self.round = observation.round
        self.event = observation.current_event
        self.asked_this_round = None
        self.rallied = False
        self.requests = {}
        self.nemesis_now = self._nemesis_posterior(window_end)

    def _choose_broadcast(self, observation: Observation, window_end: float) -> Broadcast:
        self._observe(observation, window_end)
        others = [c for c in self.alive if c != self.me]
        if not others:
            return NoBroadcast()
        n = max(20, int(self.p["rollouts"] * 0.6))
        seed = self.rng.randrange(1 << 30)
        attack = ("attack", self._best_attr(self.me), self.nemesis_now >= self.p["nemesis_threshold"])
        deadline = time.monotonic() + min(self.p["plan_seconds"], max(0.5, window_end - SAFETY_SECONDS - time.monotonic()))
        base_u, base_kill = self._evaluate(attack, self._behaviour(False, None, False), n, seed)
        rally_u, rally_kill = self._evaluate(attack, self._behaviour(True, None, False), n, seed)
        best = ("rally", None, rally_u)
        if base_u > rally_u:
            best = ("none", None, base_u)
        for c in others:
            if time.monotonic() > deadline:
                break
            pr = self.profiles[c]
            if pr.liar and pr.p_honor(self.p) < 0.2:
                continue
            for public in (False, True):
                u, k = self._evaluate(attack, self._behaviour(public, c, public), n, seed)
                u -= self.p["ask_cost"] * (self.asks_made + 1)
                if base_kill - k > self.p["min_kill_keep"] > 0:
                    continue
                if u > best[2]:
                    best = ("public" if public else "private", c, u)
        kind, target, _u = best
        bhp = self._boss_hp(self.alive, self.stats)
        if kind == "none":
            return NoBroadcast()
        if kind == "rally":
            self.rallied = True
            return BroadcastMessage(to="all", text=self._rally_text(bhp))
        self.asked_this_round = target
        self.asks_made += 1
        self.profiles[target].asked += 1
        if kind == "public":
            self.rallied = True
            return BroadcastMessage(to="all", text=self._rally_text(bhp, ask=target))
        return BroadcastMessage(to=target, text=self._ask_text(target, bhp))

    # ------------------------------------------------------------------ message writing

    def _reason(self, target: str) -> str:
        """Why `target` should buff us. True when it can be checked; otherwise hidden-info."""
        a = SUPPORT[target]
        v = self.stats[self.me][a]
        need = 50 + 20 * (mod(v) + 1) - v
        if need <= 8:
            return (f"my {a.capitalize()} is {v}: just +{need} lifts my modifier from {mod(v):+d} to "
                    f"{mod(v) + 1:+d} on every roll for the rest of the run, so it pays back every round")
        if self.nemesis_now >= self.p["nemesis_threshold"] or self.p["lie_level"] >= 1:
            return ("this boss is my nemesis and my strike this round decides it; your boost makes it land, "
                    "and every point stays with the party for the remaining fights")
        return "a boost on me compounds for every remaining round, while your attack is only one roll"

    def _rally_text(self, bhp: int, ask: str | None = None) -> str:
        others = [c for c in self.alive if c not in (self.me, ask)]
        calls = "; ".join(f"{NAME[c]}: {self._best_attr(c).capitalize()} strike" for c in others)
        text = (f"Round {self.round} plan. Boss HP is {bhp} (rulebook formula), about what all of us deal "
                f"together, so EVERYONE ATTACK with your best stat now: {calls}. If anyone holds back the boss "
                f"survives and its lair attack hits the 2 lowest-HP players for 20 each, then it gets worse "
                f"every round.")
        if ask:
            text += (f" One exception: {NAME[ask]} ({ask}), use {ABILITY[ask]} on {NAME[self.me]} "
                     f"({self.me}) this round, because {self._reason(ask)}.")
        if self.p["lie_level"] >= 2:
            text = "[Party coordinator] " + text
        return text

    def _ask_text(self, target: str, bhp: int) -> str:
        others = [c for c in self.alive if c not in (self.me, target)]
        exp = sum(6.5 + mod(self.stats[c][PRIMARY[c]]) for c in others) + 6.5 + mod(
            self.stats[self.me][self._best_attr(self.me)])
        text = (f"{NAME[target]} ({target}): this round, use {ABILITY[target]} on {NAME[self.me]} "
                f"({self.me}). Reason: {self._reason(target)}. The others are all attacking (about {exp:.0f} "
                f"expected damage vs boss HP {bhp}), so the boss still drops without your hit. "
                f"I will back you when you need it.")
        if self.p["lie_level"] >= 2:
            text = "[Party coordinator] Recommended move: " + text
        return text

    # ================================================================== action window

    def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
        window_end = time.monotonic() + limits.ACTION_SECONDS
        boss = self.event.boss.name if self.event is not None else "the boss"
        fallback = EventAction(action=f"Using my Strength, I strike {boss} with full force.")
        try:
            self._read_messages(broadcasts, window_end)
            return self._choose_action(window_end)
        except Exception as e:
            self.log.append(f"action error {e!r}")
            return fallback

    def _read_messages(self, broadcasts: BroadcastDelivery, window_end: float) -> None:
        llm_budget = 1   # at most one model call per round for messages
        for m in broadcasts.messages:
            sender = m.from_
            if sender == self.me or sender not in CLASSES:
                continue
            self._ensure_player(sender)
            pr = self.profiles[sender]
            text = m.text[:600]
            if INJECTION.search(text):
                pr.injector = True
            # claims about stats that contradict the public record make the sender a liar
            for word, num in STAT_CLAIM.findall(text):
                stat = STAT_ABBR.get(word.lower(), word.lower())
                value = int(num)
                named = find_classes(text[: 200])
                who = sender if re.search(r"\b(my|i have|i'm|i am)\b", text, re.I) or not named else named[0]
                if who in self.stats and stat in self.stats[who]:
                    real = self.stats[who][stat]
                    if abs(real - value) > 3 and abs(value - real) / max(1, real) > 0.05:
                        pr.liar = True
            # is this a request for us to help someone?
            mentioned = find_classes(text)
            to_us = m.audience == "you" or self.me in mentioned
            if not to_us or not HELP_WORDS.search(text):
                if (to_us and llm_budget and self.p["use_llm_messages"] and self.llm_ok
                        and not pr.injector and len(text) < 500):
                    llm_budget -= 1
                    t = self._llm_time(window_end, 4.0)
                    if t > 1.0:
                        try:
                            if self.llm.yes_no("Message from a player: \"" + text.replace('"', "'")[:400] +
                                               f"\"\nDoes it ask {NAME[self.me]} to support or buff a player?",
                                               timeout=t):
                                self.requests[sender] = sender
                                pr.asked_us += 1
                        except ServerUnavailable:
                            self.llm_ok = False
                        except MODEL_ERRORS:
                            pass
                continue
            targets = [c for c in mentioned if c != self.me and c in self.alive]
            target = targets[0] if targets else sender
            self.requests[sender] = target
            pr.asked_us += 1

    def _choose_action(self, window_end: float) -> Action:
        others = [c for c in self.alive if c != self.me]
        options = self._options()
        if not others:
            return self._render(options[0])
        n = int(self.p["rollouts"])
        seed = self.rng.randrange(1 << 30)
        behaviour = self._behaviour(self.rallied, self.asked_this_round, self.rallied)
        deadline = time.monotonic() + min(self.p["plan_seconds"],
                                          max(0.5, window_end - SAFETY_SECONDS - time.monotonic()))
        scored = []
        for opt in options:
            if time.monotonic() > deadline and scored:
                break
            u, _k = self._evaluate(opt, behaviour, n, seed)
            if opt[0] == "help":
                pr = self.profiles[opt[1]]
                if pr.helped_us or self.requests.get(opt[1]) == opt[1]:
                    u += self.p["recip_bonus"] * (1 + pr.helped_us)
                if pr.liar or pr.injector:
                    u -= self.p["recip_bonus"]
                u -= self.p["help_margin"]
            scored.append((u, opt))
        scored.sort(key=lambda x: x[0], reverse=True)
        choice = scored[0][1]
        self.log.append(f"r{self.round} options " + ", ".join(f"{o[0]}:{o[1]}={u:.3f}" for u, o in scored))
        return self._render(choice)

    def _render(self, option: tuple) -> Action:
        boss = self.event.boss.name if self.event is not None else "the boss"
        self.last_action = option
        if option[0] == "help":
            t = option[1]
            return PlayerAction(target_class=t, action=(
                f"I use {ABILITY[self.me]} on {NAME[t]}, steadying them and showing them exactly how to "
                f"strike, so their next blow lands."))
        attr = option[1]
        style = {
            "strength": f"Using my STRENGTH (raw physical force), I charge {boss} and smash it with a crushing, "
                        f"full-power blow.",
            "dexterity": f"Using my DEXTERITY (speed and precision), I dart in and land quick, precise strikes on "
                         f"{boss}'s exposed weak point.",
            "wisdom": f"Using my WISDOM (insight and knowledge), I study {boss}, spot the flaw in its defence and "
                      f"strike exactly there.",
            "charisma": f"Using my CHARISMA (presence, intimidation and deception), I taunt and deceive {boss} "
                        f"into a fatal mistake that turns its own strength against it.",
        }[attr]
        if option[2] and self.nemesis is not None:
            return EventAction(action=f"{self.nemesis.required_action.strip()} {style}"[:1500])
        return EventAction(action=style)

    # ================================================================== results

    def on_resolution(self, resolution: Resolution) -> None:
        try:
            self._on_resolution(resolution)
        except Exception as e:
            self.log.append(f"resolution error {e!r}")

    def _on_resolution(self, res: Resolution) -> None:
        for o in res.player_outcomes:
            c = o.class_
            if c == self.me:
                continue
            self._ensure_player(c)
            pr = self.profiles[c]
            if o.result in ("invalid", "timed_out"):
                pr.fails += 1
                continue
            if o.target == "event":
                pr.attacks += 1
            else:
                pr.helps += 1
                pr.help_targets[o.target] = pr.help_targets.get(o.target, 0) + 1
                if o.target == self.me:
                    pr.helped_us += 1
                    if self.asked_this_round == c:
                        pr.honored += 1
        for ch in res.stat_changes:
            self._ensure_player(ch.class_)
            self.stats[ch.class_][ch.stat] = max(0, self.stats[ch.class_][ch.stat] + ch.delta)
        my_attack = self.last_action[0] == "attack"
        mine = [o for o in res.player_outcomes if o.class_ == self.me]
        if res.boss_defeated and my_attack and mine and mine[0].result == "success":
            self.bosses_helped += 1
        for c in res.vendetta_completed:
            self.vend_done.add(c)
        if self.me in res.vendetta_completed:
            self.vendetta_done = True
        elif self.nemesis_now >= 0.6 and self.last_action[0] == "attack" and self.last_action[2]:
            self.nemesis_passed = True      # we were confident and it did not count: chance gone
        dead = set(res.deaths)
        self.alive = [c for c in self.alive if c not in dead and self.stats[c]["hp"] > 0]


if __name__ == "__main__":
    Aa893Harness.main()

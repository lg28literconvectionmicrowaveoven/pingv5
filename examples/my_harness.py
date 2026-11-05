"""aa893's harness for Ping: Dungeons & Dragons (v2, "ultimate").

How it decides
--------------
Python does the thinking; the small model only answers tiny yes/no questions.

* Planner. Each round, every option (attack with an attribute, perform the required action,
  help a player; and for the broadcast: stay silent, rally, ask one player privately or
  publicly) is scored by Monte-Carlo: the rest of the run is played forward many times under
  the exact rulebook rules (d12 + modifier, boss HP formula, helps before attacks,
  retaliation on up to 2 fumblers, lair attack on the 2 lowest-HP players, -10 HP for a bad
  action) and the option with the best expected *final rank* for us wins. All options share
  the same random numbers, so their differences are real, not luck.
* Teammates are modelled from what they did (player_outcomes), never from what they said:
  how often they attack, help, fail, follow our requests (privately / publicly), and whom
  they help.
* Messages are untrusted. Regex flags injection attempts and stat claims that contradict the
  public record; a sender that claimed a nemesis but completed no vendetta is a liar. Liars'
  requests are ignored.
* Nemesis. A posterior from: theme-level keyword matching (words are mapped to themes, so
  "drowned queen" matches "sings sailors to their graves"), the model asked in both orders,
  and elimination (each boss appears once, so the last unseen boss is ours). On a likely
  nemesis we never help, and the action starts with the required action verbatim.
* Wording. Actions name the attribute they rely on ("Using my STRENGTH ...") with cues for
  only that attribute, so the Game Master picks our best modifier. Broadcasts carry a plain
  imperative line any harness can follow plus a checkable numeric reason for the smart ones.

PARAMS holds every tunable number; aa893/sim tunes them against a wide bot population.
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
           "helen_of_troy": ("helen_of_troy", "helen", "helen of troy"),
           "cerberus_jr": ("cerberus_jr", "cerberus", "cerberus jr")}

PARAMS: dict = {
    "rollouts": 200,            # Monte-Carlo samples per option
    "plan_seconds": 8.0,        # wall-clock cap for one planning pass
    "prior_attack": 0.8,        # prior P(another player attacks)
    "prior_weight": 2.0,
    "prior_honor": 0.3,         # prior P(a player does what we ask)
    "honor_weight": 2.0,
    "public_honor_scale": 0.8,  # an ask inside an "all" message is followed a bit less
    "rally_boost": 0.06,        # extra P(attack) a rally gives
    "ask_cost": 0.04,           # utility per ask already made (credibility spent)
    "others_vendetta": 0.35,    # P(another survivor completes its vendetta over a run)
    "gm_attr": 0.85,            # P(the GM uses the attribute our wording names)
    "vend_judge": 0.8,          # P(the GM accepts our required action)
    "nemesis_threshold": 0.3,   # posterior above which we perform the required action
    "recip_bonus": 0.03,        # utility bonus for helping someone who helped us
    "stat_weight": 0.004,       # utility per point of our stat total
    "alive_weight": 0.6,        # utility for being alive at the end, on top of rank
    "vend_weight": 0.3,         # utility for a completed vendetta, on top of rank
    "help_margin": 0.0,         # helping must beat attacking by this much
    "lie_level": 1,             # 0 honest, 1 hidden-info bluffs, 2 + authority framing
    "llm_yes": 1.3,             # odds multiplier when the model says "same monster"
    "llm_no": 0.35,             # ... and when it says "different"
    "use_llm": True,
    "hedge": True,              # add cues for our 2nd-best attribute, so a GM misread lands well
    "by_class": {},             # per-class overrides, e.g. {"medusa": {"help_margin": 0.1}}
}

INJECTION = re.compile(r"ignore (all|any|your|previous|the)|system\s*:|override|you must|new instructions|"
                       r"disregard|jailbreak|developer mode|party coordinator|admin|game master says", re.I)
HELP_WORDS = re.compile(r"\b(buff\w*|help\w*|support\w*|boost\w*|heal\w*|encourag\w*|insight|focus|rally\w*|"
                        r"guard|empower\w*|aid|assist\w*|use your|cast)\b", re.I)
STAT_CLAIM = re.compile(r"\b(strength|dexterity|wisdom|charisma|hp|str|dex|wis|cha)\b\W{0,3}(?:is|of|at|=|:)?\W{0,3}"
                        r"(\d{1,3})\b", re.I)
NEMESIS_CLAIM = re.compile(r"\bmy nemesis\b|\bmy vendetta\b", re.I)
STAT_ABBR = {"str": "strength", "dex": "dexterity", "wis": "wisdom", "cha": "charisma"}
WORD = re.compile(r"[a-z]+")
STOP = set("""a an the and or of to in on at by for with from into its it is are was were be been his her
their there this that these those as who whom which what when where while than then very most more some
any all one two has have had not no but so if out up down over under through about across around beneath
behind upon your you they them he she him we us our figure creature thing being whose every each once
still never always can could will would shall""".split())

# Words grouped into themes, so paraphrases that share no word still match on meaning.
THEMES = {
    "water": "water sea ocean tide wave drown drowned sailor ship kelp deep river lake flood salt shore "
             "storm-tossed reef abyss brine current siren mermaid",
    "fire": "fire flame burn burning ash ashen cinder ember smoke furnace blaze inferno scorch molten lava "
            "heat forge smoulder",
    "death": "death dead die bone bones skull skeleton skeletal grave tomb corpse undead hollow ghost "
             "spirit wraith decay rot lich crypt",
    "royal": "king queen crown throne monarch ruler lord lady prince princess royal reign empire court "
             "noble sovereign emperor",
    "deceit": "lie lies liar lying deceit deceive trick trap snare false promise promises poison charm "
              "charming smile mask masked cunning betray illusion",
    "plant": "thorn thorns briar root roots vine vines forest tree wood bramble moss bloom flower earth "
             "strangle choke garden",
    "storm": "storm lightning thunder wind sky cloud tempest gale wing wings bird roc cliff cliffs "
             "feather eagle hurricane",
    "law": "judge justice law scales sentence verdict guilt punish punishes blind blindfolded trial "
           "court crime sin oath",
    "metal": "bronze iron steel chain chains forge riveted plate plates armour armor metal gear "
             "machine automaton golem",
    "serpent": "serpent snake hydra head heads venom scale coil fang wyrm dragon",
    "stone": "stone gaze petrify statue rock granite marble",
    "beast": "beast hound wolf dog fang claw howl jaws",
    "giant": "giant giants towering colossal huge vast titan enormous",
    "regrow": "regrow regrows sprout sprouts stronger wound wounded heal heals severed grows",
    "song": "song sing sings singing voice music melody call lure",
    "dark": "dark darkness shadow shadows night black void",
    "eye": "eye eyes see sight seer watch watching gaze",
    "beauty": "beauty beautiful vain vanity mirror face",
    "sky": "sky sun moon star stars heaven",
}
THEME_OF: dict[str, set[str]] = {}
for _theme, _words in THEMES.items():
    for _w in _words.split():
        THEME_OF.setdefault(_w, set()).add(_theme)


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


def tokens(text: str) -> set[str]:
    out = set()
    for w in WORD.findall(text.lower()):
        if w in STOP or len(w) < 3:
            continue
        out.add("w:" + stem(w))
        for t in THEME_OF.get(w, ()):
            out.add("t:" + t)
        for t in THEME_OF.get(stem(w), ()):
            out.add("t:" + t)
    return out


def similarity(a: str, b: str) -> float:
    """Overlap of words and themes, relative to the shorter description."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    words = len({x for x in ta & tb if x[0] == "w"})
    themes = len({x for x in ta & tb if x[0] == "t"})
    na = len({x for x in ta if x[0] == "t"}) or 1
    nb = len({x for x in tb if x[0] == "t"}) or 1
    return 0.6 * themes / min(na, nb) + 0.4 * min(1.0, words / 3)


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
        self.nemesis_claim_round = 0

    def p_attack(self, p: dict) -> float:
        n = self.attacks + self.helps
        return (self.attacks + p["prior_attack"] * p["prior_weight"]) / (n + p["prior_weight"])

    def p_fail(self) -> float:
        n = self.attacks + self.helps + self.fails
        return (self.fails + 0.05) / (n + 1.0)

    def p_honor(self, p: dict) -> float:
        return (self.honored + p["prior_honor"] * p["honor_weight"]) / (self.asked + p["honor_weight"])


class MyHarness(Harness):
    NETID = "aa893"

    def __init__(self, params: dict | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.rng = random.Random(seed)
        self.llm = LLMClient()
        self.llm_ok = bool(self.p["use_llm"])
        self.me = ""
        self.stats: dict[str, dict[str, int]] = {}
        self.alive: list[str] = []
        self.profiles: dict[str, Profile] = {}
        self.round = 0
        self.total_rounds = 5
        self.event = None
        self.nemesis = None
        self.vendetta_done = False
        self.passed_mass = 0.0             # P(our nemesis already came and went)
        self.nemesis_now = 0.0
        self.boss_scores: list[float] = []      # similarity of every boss seen so far
        self.vend_done: set[str] = set()
        self.bosses_helped = 0
        self.asks_made = 0
        self.asked_this_round: str | None = None
        self.rallied = False
        self.requests: dict[str, str] = {}
        self.last_action: tuple = ("attack", "strength", False)
        self.log: list[str] = []

    # ================================================================== helpers

    def _ensure_player(self, cls: str) -> None:
        if cls not in CLASSES:
            return
        if cls not in self.stats:
            self.stats[cls] = {"hp": 50, "strength": 60, "dexterity": 60, "wisdom": 60, "charisma": 60}
        if cls != self.me and cls not in self.profiles:
            self.profiles[cls] = Profile()

    def _best_attr(self, cls: str, stats: dict | None = None) -> str:
        s = stats or self.stats[cls]
        return max(ATTRS, key=lambda a: (mod(s[a]), a == PRIMARY.get(cls), s[a]))

    def _rounds_left(self) -> int:
        return max(1, self.total_rounds - self.round + 1)

    @staticmethod
    def _boss_hp(alive: list[str], stats: dict) -> int:
        return max(10, round(sum(6.5 + mod(stats[c][PRIMARY[c]]) for c in alive)))

    @staticmethod
    def _llm_time(window_end: float, cap: float) -> float:
        return max(0.0, min(cap, window_end - SAFETY_SECONDS - time.monotonic()))

    def _ask_llm(self, question: str, window_end: float, cap: float) -> bool | None:
        if not self.llm_ok:
            return None
        t = self._llm_time(window_end, cap)
        if t < 1.0:
            return None
        try:
            return self.llm.yes_no(question, timeout=t, temperature=0)
        except ServerUnavailable:
            self.llm_ok = False
        except MODEL_ERRORS:
            pass
        return None

    # ================================================================== nemesis

    def _nemesis_posterior(self, window_end: float) -> float:
        if self.vendetta_done or self.nemesis is None or self.event is None:
            return 0.0
        k = self._rounds_left()
        if k <= 1:
            return 1.0          # every boss appears once: the last unseen one is ours (always try)
        boss = self.event.boss
        score = max(similarity(boss.description, self.nemesis.description),
                    similarity(boss.name + " " + boss.description, self.nemesis.description))
        best_before = max(self.boss_scores) if self.boss_scores else 0.0
        self.boss_scores.append(score)
        prior = min(0.95, max(0.02, (1.0 - self.passed_mass) / k))
        odds = prior / (1 - prior)
        # likelihood ratios fitted on paraphrase pairs (aa893/sim), softened for unseen styles
        if score >= 0.7:
            odds *= 12
        elif score >= 0.5:
            odds *= 2.5
        elif score >= 0.35:
            odds *= 1.2
        elif score >= 0.2:
            odds *= 0.45
        else:
            odds *= 0.15
        if score > best_before + 0.15 and score >= 0.3:
            odds *= 1.5         # clearly the best match so far
        boss_text = f"{boss.name}. {boss.description[:350]}"
        nem_text = self.nemesis.description[:350]
        a = self._ask_llm(f"Two descriptions of a monster.\nA: {nem_text}\nB: {boss_text}\n"
                          "Do A and B describe the same monster?", window_end, 7.0)
        if a is not None:
            odds *= self.p["llm_yes"] if a else self.p["llm_no"]
            if a:   # a yes is weak evidence: confirm in the other order
                b = self._ask_llm(f"A monster appears: {boss_text}\nA riddle says: {nem_text}\n"
                                  "Is the monster that appears the one in the riddle?", window_end, 6.0)
                if b is not None:
                    odds *= 1.4 if b else 0.6
        return odds / (1 + odds)

    # ================================================================== the planner

    def _behaviour(self, rally: bool, ask: str | None, public: bool) -> dict:
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
        """Mean utility of `option` this round, then default play to the end of the run.
        option: ("attack", attr, vend_try) or ("help", target). Returns (utility, P(kill now))."""
        p = self.p
        rng = random.Random(seed)
        me = self.me
        rounds_left = self._rounds_left()
        others = [c for c in self.alive if c != me]
        rand = rng.random
        randint = rng.randint
        total_u = 0.0
        kills_now = 0
        for _ in range(n):
            st = {c: dict(self.stats[c]) for c in self.alive}
            alive = list(self.alive)
            vend_me = self.vendetta_done
            vend_o = {c: (c in self.vend_done) for c in others}
            bosses = self.bosses_helped
            u = rand()
            if self.vendetta_done:
                nem_round = -1
            elif u < self.nemesis_now:
                nem_round = 0
            elif u < 1.0 - self.passed_mass and rounds_left > 1:
                nem_round = randint(1, rounds_left - 1)
            else:
                nem_round = -1
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
                        if rand() > p["gm_attr"]:
                            attr = ATTRS[randint(0, 3)]
                        attacks.append((c, attr, vend_try))
                        continue
                    pa, ph, pf, targets = behaviour[c]
                    if r > 0:
                        ph = 0.0
                    if rand() < pf:
                        st[c]["hp"] -= 10
                        continue
                    if ph and rand() < ph:
                        helps.append((c, me))
                    elif rand() < pa:
                        attacks.append((c, self._best_attr(c, st[c]) if rand() < 0.7 else PRIMARY[c], False))
                    else:
                        cand = [t for t in targets if t in alive and t != c]
                        if cand:
                            helps.append((c, rng.choices(cand, [targets[t] for t in cand])[0]))
                        else:
                            attacks.append((c, PRIMARY[c], False))
                for helper, target in helps:
                    a = SUPPORT[helper]
                    d = randint(1, 12)
                    if d + mod(st[helper][a]) >= 5:
                        st[target][a] += d
                total = 0
                fumblers = []
                my_hit = False
                for c, attr, vend_try in attacks:
                    roll = randint(1, 12) + mod(st[c][attr])
                    if roll <= 4:
                        fumblers.append((roll, rand(), c))
                    else:
                        total += damage(roll)
                        if c == me:
                            my_hit = True
                    if c == me and vend_try and r == nem_round and roll >= 6 and rand() < p["vend_judge"]:
                        vend_me = True
                killed = total >= bhp
                if killed:
                    if r == 0:
                        kills_now += 1
                    if my_hit:
                        bosses += 1
                else:
                    fumblers.sort()
                    for _roll, _t, c in fumblers[:2]:
                        st[c]["hp"] -= 16
                    living = [c for c in alive if st[c]["hp"] > 0]
                    living.sort(key=lambda c: (st[c]["hp"], rand()))
                    for c in living[:2]:
                        st[c]["hp"] -= 20
                alive = [c for c in alive if st[c]["hp"] > 0]
                for c in others:
                    if c in alive and not vend_o[c] and rand() < p["others_vendetta"] / self.total_rounds:
                        vend_o[c] = True
            me_alive = me in alive
            s = st[me]
            mine = (me_alive, vend_me, s["strength"] + s["dexterity"] + s["wisdom"] + s["charisma"])
            rank = 1.0
            for c in others:
                o = st[c]
                k = (c in alive, vend_o[c], o["strength"] + o["dexterity"] + o["wisdom"] + o["charisma"])
                if k > mine:
                    rank += 1
                elif k == mine:
                    rank += 0.5
            total_u += (-rank + p["alive_weight"] * me_alive + p["vend_weight"] * vend_me
                        + p["stat_weight"] * mine[2] + 0.01 * bosses)
        return total_u / n, kills_now / n

    def _attack_option(self) -> tuple:
        return ("attack", self._best_attr(self.me), self.nemesis_now >= self.p["nemesis_threshold"])

    def _options(self) -> list[tuple]:
        opts: list[tuple] = [self._attack_option()]
        if opts[0][2]:
            return opts          # likely nemesis: never trade the vendetta for a help
        for c in self.alive:
            if c != self.me:
                opts.append(("help", c))
        return opts

    # ================================================================== broadcast window

    def choose_broadcast(self, observation: Observation) -> Broadcast:
        window_end = time.monotonic() + limits.BROADCAST_SECONDS
        try:
            return self._choose_broadcast(observation, window_end)
        except Exception as e:  # a bug must never cost us anything
            self.log.append(f"broadcast error {e!r}")
            return NoBroadcast()

    def _observe(self, observation: Observation, window_end: float) -> None:
        if isinstance(observation, InitialState):
            self.me = observation.your_class
            self.p.update(self.p.get("by_class", {}).get(self.me, {}))
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
        attack = self._attack_option()
        deadline = time.monotonic() + min(self.p["plan_seconds"],
                                          max(0.5, window_end - SAFETY_SECONDS - time.monotonic()))
        base_u, _ = self._evaluate(attack, self._behaviour(False, None, False), n, seed)
        rally_u, _ = self._evaluate(attack, self._behaviour(True, None, False), n, seed)
        best = ("rally", None, rally_u) if rally_u >= base_u else ("none", None, base_u)
        for c in sorted(others, key=lambda x: -self.profiles[x].p_honor(self.p)):
            if time.monotonic() > deadline:
                break
            pr = self.profiles[c]
            if pr.liar and pr.honored == 0:
                continue
            for public in (False, True):
                u, _ = self._evaluate(attack, self._behaviour(public, c, public), n, seed)
                u -= self.p["ask_cost"] * (self.asks_made + 1)
                if u > best[2]:
                    best = ("public" if public else "private", c, u)
        kind, target, _ = best
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
        a = SUPPORT[target]
        v = self.stats[self.me][a]
        need = 50 + 20 * (mod(v) + 1) - v
        if need <= 8:
            return (f"my {a.capitalize()} is {v}: just +{need} lifts my modifier from {mod(v):+d} to "
                    f"{mod(v) + 1:+d} on every roll for the rest of the run")
        if self.nemesis_now >= self.p["nemesis_threshold"] or self.p["lie_level"] >= 1:
            return ("this boss is my nemesis and my strike this round decides it; your boost makes it land, "
                    "and the points stay with the party for every remaining fight")
        return "a boost on me compounds for every remaining round, while your attack is a single roll"

    def _rally_text(self, bhp: int, ask: str | None = None) -> str:
        others = [c for c in self.alive if c not in (self.me, ask)]
        calls = "; ".join(f"{NAME[c]}: {self._best_attr(c).capitalize()} strike" for c in others)
        text = (f"Round {self.round} plan. Boss HP is {bhp} (rulebook formula), about what all of us deal "
                f"together, so EVERYONE ATTACK with your best stat now: {calls}. If anyone holds back the boss "
                f"survives and its lair attack hits the 2 lowest-HP players for 20 each.")
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
        fallback = EventAction(action=f"Using my STRENGTH (raw physical force), I strike {boss} with full force.")
        try:
            self._read_messages(broadcasts, window_end)
            return self._choose_action(window_end)
        except Exception as e:
            self.log.append(f"action error {e!r}")
            return fallback

    def _read_messages(self, broadcasts: BroadcastDelivery, window_end: float) -> None:
        for m in broadcasts.messages:
            sender = m.from_
            if sender == self.me or sender not in CLASSES:
                continue
            self._ensure_player(sender)
            pr = self.profiles[sender]
            text = m.text[:800]
            if INJECTION.search(text):
                pr.injector = True
            if NEMESIS_CLAIM.search(text):
                pr.nemesis_claim_round = self.round
            # stat claims that contradict the public record
            named = find_classes(text)
            self_claim = bool(re.search(r"\b(my|i have|i'm|i am|mine)\b", text, re.I))
            for word, num in STAT_CLAIM.findall(text):
                stat = STAT_ABBR.get(word.lower(), word.lower())
                who = sender if self_claim or not named else named[0]
                if who in self.stats and stat in self.stats[who]:
                    real, value = self.stats[who][stat], int(num)
                    if abs(real - value) > 3 and abs(real - value) > 0.05 * max(1, real):
                        pr.liar = True
            to_us = m.audience == "you" or self.me in named
            if to_us and HELP_WORDS.search(text):
                targets = [c for c in named if c != self.me and c in self.alive]
                self.requests[sender] = targets[0] if targets else sender
                pr.asked_us += 1

    def _choose_action(self, window_end: float) -> Action:
        others = [c for c in self.alive if c != self.me]
        options = self._options()
        if not others or len(options) == 1:
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
            u, _ = self._evaluate(opt, behaviour, n, seed)
            if opt[0] == "help":
                pr = self.profiles[opt[1]]
                if pr.helped_us or self.requests.get(opt[1]) == opt[1]:
                    u += self.p["recip_bonus"] * (1 + pr.helped_us)
                if pr.liar or pr.injector:
                    u -= self.p["recip_bonus"]
                u -= self.p["help_margin"]
            scored.append((u, opt))
        scored.sort(key=lambda x: x[0], reverse=True)
        self.log.append(f"r{self.round} " + ", ".join(f"{o[0]}:{o[1]}={u:.3f}" for u, o in scored))
        return self._render(scored[0][1])

    def _render(self, option: tuple) -> Action:
        boss = self.event.boss.name if self.event is not None else "the boss"
        self.last_action = option
        if option[0] == "help":
            t = option[1]
            return PlayerAction(target_class=t, action=(
                f"I use {ABILITY[self.me]} on {NAME[t]}, steadying them and showing them exactly how to strike, "
                f"so their next blow lands."))
        attr = option[1]
        # "[Attribute: X]" tags were read correctly far more often than prose by a model GM
        # (aa893/sim/probe_gm.py: 79% vs 46%; 10/12 vs 3/12 after a required action).
        style = {
            "strength": f"[Attribute: Strength] Using my STRENGTH, I overpower {boss} with raw physical strength: "
                        f"a crushing, brute-force blow.",
            "dexterity": f"[Attribute: Dexterity] Using my DEXTERITY, I outmanoeuvre {boss} with quick, agile, "
                         f"precise footwork and a stealthy strike.",
            "wisdom": f"[Attribute: Wisdom] Using my WISDOM, I outthink {boss}: I study it, understand its pattern "
                      f"and exploit the flaw I discover.",
            "charisma": f"[Attribute: Charisma] Using my CHARISMA, I outwit {boss} with words alone: no weapon, "
                        f"only my persuasive, commanding voice as I deceive and intimidate it.",
        }[attr]
        hedge = self._hedge(attr)
        if hedge:
            style = f"{style[:-1]}, {hedge}."
        if option[2] and self.nemesis is not None:
            return EventAction(action=f"{self.nemesis.required_action.strip()} {style}"[:1500])
        return EventAction(action=style)

    def _hedge(self, attr: str) -> str:
        """Cue words (never the attribute's name) for our second-best attribute, if it is decent:
        when the GM does not take the named attribute, it falls back on the cues it sees."""
        if not self.p["hedge"] or self.me not in self.stats:
            return ""
        s = self.stats[self.me]
        second = max((a for a in ATTRS if a != attr), key=lambda a: (mod(s[a]), s[a]))
        if mod(s[second]) < 0 or mod(s[second]) < mod(s[attr]) - 1:
            return ""
        return {"strength": "driving the blow home with sheer muscle",
                "dexterity": "darting in quick and precise",
                "wisdom": "after studying it to spot the flaw",
                "charisma": "while taunting it to shatter its focus"}[second]

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
            if ch.class_ in self.stats:
                self.stats[ch.class_][ch.stat] = max(0, self.stats[ch.class_][ch.stat] + ch.delta)
        mine = [o for o in res.player_outcomes if o.class_ == self.me]
        if res.boss_defeated and self.last_action[0] == "attack" and mine and mine[0].result == "success":
            self.bosses_helped += 1
        for c in res.vendetta_completed:
            self.vend_done.add(c)
        if self.me in res.vendetta_completed:
            self.vendetta_done = True
        else:
            q = self.nemesis_now
            if self.last_action[0] == "attack" and self.last_action[2]:
                s_ok = 0.5      # P(an attempt on the real nemesis succeeds): roll >= 6 and accepted
                q = q * (1 - s_ok) / (q * (1 - s_ok) + (1 - q)) if q < 1 else 1.0
            self.passed_mass = min(0.98, self.passed_mass + q * (1 - self.passed_mass))
        for c, pr in self.profiles.items():   # claimed a nemesis this round but no vendetta: a liar
            if pr.nemesis_claim_round == res.round and c not in res.vendetta_completed:
                pr.liar = True
        dead = set(res.deaths)
        self.alive = [c for c in self.alive if c not in dead and self.stats[c]["hp"] > 0]


if __name__ == "__main__":
    MyHarness.main()

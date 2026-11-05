"""AI Harness for Ping - Dungeons and Dragons.

Design:
  1. NETID set to "mg163@snu.edu.in".
  2. Compact RAM memory: round outcomes, stat changes, deaths, vendettas, teammate HP.
  3. Nemesis detection: pure-Python stem overlap (no LLM). A false positive is almost free
     (the action still rolls and deals normal damage), a miss costs the vendetta, so the
     threshold is low and the attempt is forced from round 4 until our own vendetta is
     confirmed completed in a Resolution.
  4. Broadcast: round 1 intro, low-HP warning; otherwise silence.
  5. Action priority:
       a) Nemesis round -> required_action near-verbatim.
       b) Rounds 1-2 only, healthy, strong synergy -> help a teammate (permanent boost).
       c) Otherwise attack with primary-attribute-themed text (templates; optional LLM flavour).
  6. Every path has a valid fallback; choose_action never raises.
"""
from __future__ import annotations

import math
import re
import time

from harness_api import Harness, LLMClient, limits
from harness_api.messages import (
    Action,
    BroadcastDelivery,
    BroadcastMessage,
    EventAction,
    InitialState,
    NoBroadcast,
    PlayerAction,
    Resolution,
    Stats,
    TurnMessage,
)

MODEL_ERRORS = (Exception,)   # any model failure falls back to templates
SAFETY_SECONDS = 5.0

USE_LLM_FLAVOUR = False       # templates are safer; set True to let the model vary attack text
HELP_UNTIL_ROUND = 0          # This is better for a selfish playstyle
HELP_MIN_HP = 20              # do not help when our own HP is at or below this
FORCE_VENDETTA_FROM = 3       # from this round, always attempt the vendetta until completed
STEM_THRESHOLD = 0.15         # overlap needed to attempt the vendetta before the forced rounds

ATTRS = ("strength", "dexterity", "wisdom", "charisma")

# ------------------------------------------------------------------ class data

CLASS_INFO: dict[str, dict[str, str]] = {
    "francesca":     {"primary": "charisma",  "support_raises": "charisma",  "support_name": "hearts_encouragement"},
    "aristotle":     {"primary": "wisdom",    "support_raises": "wisdom",    "support_name": "philosophical_insight"},
    "medusa":        {"primary": "dexterity", "support_raises": "dexterity", "support_name": "petrifying_focus"},
    "helen_of_troy": {"primary": "strength",  "support_raises": "strength",  "support_name": "rallying_presence"},
    "cerberus_jr":   {"primary": "dexterity", "support_raises": "strength",  "support_name": "pack_guard"},
}

TEMPLATES = {
    "strength":  "I charge {boss} and smash it with a crushing blow of raw brute force.",
    "dexterity": "I dart around {boss}, fast and silent, and strike its weak point with precision.",
    "wisdom":    "I study {boss}, use my insight and knowledge to spot its weakness, and exploit it.",
    "charisma":  "I stare down {boss} and intimidate it with a commanding, persuasive speech.",
}

KEYWORDS = {
    "strength":  ("smash", "force", "brute", "crush", "strength", "power", "heave", "slam", "strike"),
    "dexterity": ("dart", "precise", "precision", "swift", "stealth", "fast", "dodge", "nimble", "agile"),
    "wisdom":    ("insight", "weakness", "study", "knowledge", "observe", "wisdom", "analyze", "weak"),
    "charisma":  ("persuade", "intimidate", "charm", "speech", "convince", "taunt", "deceive", "command"),
}

SUPPORT_TEMPLATES = {
    "hearts_encouragement":  "I bolster {target}'s confidence and social presence with heartfelt encouragement.",
    "philosophical_insight": "I help {target} understand the situation clearly, improving their judgement and perception.",
    "petrifying_focus":      "I guide {target}'s movements with petrifying focus, lending them precision and control.",
    "rallying_presence":     "I call {target} to action with rallying presence, inspiring decisive force.",
    "pack_guard":            "I stand beside {target} and help them brace and hold their ground with pack guard.",
}

STOP = {"the", "and", "that", "with", "from", "this", "into", "your", "their", "which",
        "have", "has", "was", "were", "for", "its", "but", "not", "who", "are", "been",
        "each", "they", "them", "than", "then", "will", "when", "what", "some", "over"}


# ------------------------------------------------------------------ utilities

def time_left(window_end: float, cap: float = 20.0) -> float:
    """Seconds a model call may still use, so the window is never missed."""
    return max(0.0, min(cap, window_end - SAFETY_SECONDS - time.monotonic()))


def get_modifier(attribute_value: int) -> int:
    """Attribute modifier: floor((attribute - 50) / 20)."""
    return math.floor((attribute_value - 50) / 20)


def _stems(text: str) -> set[str]:
    """Crude stemming: first 5 letters of each word longer than 3 chars, stop words removed."""
    return {w[:5] for w in re.findall(r"[a-z]+", str(text).lower())
            if len(w) > 3 and w not in STOP}


def _cap_words(text: str, n: int = 140) -> str:
    """Cut to n words and 2000 characters (the game cuts at 150 words / 2000 chars)."""
    return " ".join(str(text).split()[:n])[:2000]


# ------------------------------------------------------------------ harness

class MyHarness(Harness):
    NETID = "mg163"

    def __init__(self) -> None:
        super().__init__()
        self.llm = LLMClient()

        # --- Game state ---
        self.me = ""
        self.primary = "strength"
        self.round_no = 0
        self.event = None

        # --- Nemesis tracking ---
        self.nem_stems: set[str] = set()
        self.nem_action = ""
        self.vendetta_done = False       # True only once the game confirms OUR vendetta
        self.vendetta_now = False

        # --- Teammate tracking ---
        self.teammate_stats: dict[str, Stats] = {}
        self.dead_players: set[str] = set()
        self.vendettas_completed: set[str] = set()

        # --- Compact memory ---
        self.memory: list[str] = []

    # ------------------------------------------------------------------ nemesis

    def _is_my_nemesis(self, boss) -> bool:
        """Stem-overlap check. Low threshold early, forced from FORCE_VENDETTA_FROM."""
        if self.vendetta_done or not self.nem_action:
            return False
        if self.round_no >= FORCE_VENDETTA_FROM:
            return True
        if not self.nem_stems:
            return False
        boss_stems = _stems(f"{boss.name} {boss.description}")
        return len(self.nem_stems & boss_stems) / len(self.nem_stems) >= STEM_THRESHOLD

    # ------------------------------------------------------------------ help strategy

    def _pick_help_target(self) -> str | None:
        """Living teammate who benefits most from our support ability, or None."""
        support_stat = CLASS_INFO.get(self.me, CLASS_INFO["medusa"])["support_raises"]

        best_target, best_score = None, -1
        for cls, stats in self.teammate_stats.items():
            if cls == self.me or cls in self.dead_players:
                continue
            t_primary = CLASS_INFO.get(cls, {}).get("primary", "strength")
            score = 0
            if t_primary == support_stat:
                score += 40
            if stats.hp <= 20:
                score += 25
            elif stats.hp <= 35:
                score += 10
            stat_val = getattr(stats, support_stat, 50)
            if stat_val < 70:
                score += 15
            elif stat_val < 90:
                score += 5
            if score > best_score:
                best_score, best_target = score, cls

        return best_target if best_score >= 30 else None

    # ------------------------------------------------------------------ LLM flavour

    def _flavour(self, boss, window_end: float) -> str | None:
        """Optional attack text from the model, validated by keywords. None on any failure."""
        timeout = time_left(window_end, cap=8)
        if timeout < 1.0:
            return None
        desc = str(boss.description)[:150]
        try:
            text = self.llm.short_text(
                f"Boss: {boss.name}. {desc}\nOne short combat action using {self.primary}:",
                max_chars=200,
                timeout=timeout,
            ).strip()
        except MODEL_ERRORS:
            return None
        low = text.lower()
        if len(text) < 15 or not any(k in low for k in KEYWORDS.get(self.primary, ())):
            return None
        return text

    # ------------------------------------------------------------------ broadcast

    def choose_broadcast(self, observation: InitialState | TurnMessage) -> BroadcastMessage | NoBroadcast:
        try:
            return self._broadcast(observation)
        except Exception:
            return NoBroadcast()

    def _broadcast(self, observation) -> BroadcastMessage | NoBroadcast:
        self.round_no = observation.round
        self.event = observation.current_event

        if isinstance(observation, InitialState):
            self.me = observation.your_class
            self.nem_stems = _stems(observation.nemesis.description)
            self.nem_action = observation.nemesis.required_action
            for other in observation.other_players:
                self.teammate_stats[other.class_] = other.stats

        self.teammate_stats[self.me] = observation.your_stats
        stats_dict = {a: getattr(observation.your_stats, a) for a in ATTRS}
        self.primary = max(ATTRS, key=lambda a: stats_dict[a])

        self.vendetta_now = self._is_my_nemesis(self.event.boss)

        living_others = [c for c in self.teammate_stats
                         if c != self.me and c not in self.dead_players]
        if not living_others:
            return NoBroadcast()

        my_info = CLASS_INFO.get(self.me, CLASS_INFO["medusa"])

        if self.round_no == 1:
            text = (
                f"I am {self.me}. I specialise in {my_info['primary']}. "
                f"My support ability boosts {my_info['support_raises']}. "
                f"Let us survive together."
            )
            return BroadcastMessage(to="all", text=text)

        our_stats = self.teammate_stats.get(self.me)
        if our_stats and our_stats.hp <= 20:
            return BroadcastMessage(
                to="all",
                text=f"I am at {our_stats.hp} HP and in danger. Support would help me survive.",
            )

        return NoBroadcast()

    # ------------------------------------------------------------------ action

    def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
        window_end = time.monotonic() + limits.ACTION_SECONDS
        boss_name = "the boss"
        try:
            boss = self.event.boss
            boss_name = str(boss.name)
            return self._action(boss, window_end)
        except Exception:
            return EventAction(action=f"Attack {boss_name}.")

    def _action(self, boss, window_end: float) -> Action:
        # Priority 1: vendetta - required action, near-verbatim, full 150 words allowed
        if self.vendetta_now:
            return EventAction(action=_cap_words(self.nem_action, n=150))

        # Priority 2: early help, only when healthy and there is strong synergy
        my_stats = self.teammate_stats.get(self.me)
        my_hp = my_stats.hp if my_stats else 0
        if self.round_no <= HELP_UNTIL_ROUND and my_hp > HELP_MIN_HP:
            target = self._pick_help_target()
            if target is not None:
                support_name = CLASS_INFO.get(self.me, CLASS_INFO["medusa"])["support_name"]
                template = SUPPORT_TEMPLATES.get(support_name, "I use my support ability to help {target}.")
                return PlayerAction(target_class=target, action=template.format(target=target))

        # Priority 3: attack with the primary attribute
        text = None
        if USE_LLM_FLAVOUR:
            text = self._flavour(boss, window_end)
        if not text:
            text = TEMPLATES[self.primary].format(boss=boss.name)
        return EventAction(action=_cap_words(text))

    # ------------------------------------------------------------------ resolution

    def on_resolution(self, resolution: Resolution) -> None:
        """Record key outcomes. Must return quickly (5 s limit)."""
        try:
            self._resolve(resolution)
        except Exception:
            pass

    def _resolve(self, resolution: Resolution) -> None:
        parts = [f"R{resolution.round}: boss {'defeated' if resolution.boss_defeated else 'survived'}"]

        for dead in resolution.deaths:
            self.dead_players.add(dead)
        if resolution.deaths:
            parts.append(f"died={','.join(resolution.deaths)}")

        for v in resolution.vendetta_completed:
            self.vendettas_completed.add(v)
        if resolution.vendetta_completed:
            parts.append(f"vendetta={','.join(resolution.vendetta_completed)}")
        if self.me in resolution.vendetta_completed:
            self.vendetta_done = True

        for change in resolution.stat_changes:
            cls = change.class_
            old = self.teammate_stats.get(cls)
            if old is not None:
                d = {a: getattr(old, a) for a in ("hp",) + ATTRS}
                if change.stat in d:
                    d[change.stat] = max(0, d[change.stat] + change.delta)
                    self.teammate_stats[cls] = Stats(**d)
            if cls == self.me and change.stat == "hp":
                parts.append(f"my_hp{change.delta:+d}")

        self.memory.append(" ".join(parts))


if __name__ == "__main__":  # python my_harness_mg163.py: join a game; --check: self-check
    MyHarness.main()
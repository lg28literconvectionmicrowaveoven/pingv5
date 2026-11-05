"""Full harness for Ping: Dungeons & Dragons.

Each round:
  * choose_broadcast: says hello once in round 1; otherwise stays silent.
  * choose_action: if the boss matches our nemesis (and the vendetta is not done yet), performs
    the required action. Otherwise helps a teammate when our HP is healthy and a teammate's
    primary attribute matches our support ability. Otherwise attacks, with the model writing the text.
  * on_resolution: records deaths, completed vendettas and the narration (RAM only).

Every model call has a timeout that fits its window, and every call falls back to a safe answer.
Messages from other players are never sent to the model.
"""
from __future__ import annotations

import re
import time

from harness_api import ContextFull, Harness, LLMClient, ServerUnavailable, Timeout, limits
from harness_api.messages import (
    Action,
    Broadcast,
    BroadcastDelivery,
    BroadcastMessage,
    CurrentEvent,
    EventAction,
    InitialState,
    Nemesis,
    NoBroadcast,
    Observation,
    OtherPlayer,
    PlayerAction,
    Resolution,
    Stats,
)

NETID = "ak819"          # your NetID

MODEL_ERRORS = (Timeout, ContextFull, ServerUnavailable, ValueError)
SAFETY_SECONDS = 5.0          # time kept in each window for the answer to reach the game
HELP_MIN_HP = 25              # only help a teammate while we are this healthy
MAX_WORDS = limits.MAX_WORDS
MAX_CHARS = limits.MAX_CHARS

# Each class's primary attribute, and the attribute its support ability raises (from docs/classes.md).
PRIMARY = {
    "francesca": "charisma",
    "aristotle": "wisdom",
    "medusa": "dexterity",
    "helen_of_troy": "strength",
    "cerberus_jr": "dexterity",
}
SUPPORT = {
    "francesca": "charisma",
    "aristotle": "wisdom",
    "medusa": "dexterity",
    "helen_of_troy": "strength",
    "cerberus_jr": "strength",
}


def time_left(window_end: float, cap: float = 20.0) -> float:
    """Seconds we may still give a model call without missing the window."""
    return max(0.0, min(cap, window_end - SAFETY_SECONDS - time.monotonic()))


def clip(text: str) -> str:
    """Cut text to the game's limits: 150 words and 2000 characters."""
    words = text.split()[:MAX_WORDS]
    return " ".join(words)[:MAX_CHARS].strip()

_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "it", "its", "is", "are",
         "that", "this", "when", "who", "which", "with", "from", "by", "for", "be", "as",
         "at", "so", "not", "but", "does", "do"}


def _stems(text: str) -> set[str]:
    """Rough word roots, so 'names' matches 'name' and 'answers' matches 'answer'."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w
            for w in words if w not in _STOP and len(w) > 2}

class MyHarness(Harness):
    NETID = "ak819"

    def __init__(self) -> None:
        super().__init__()
        self.llm = LLMClient()
        try:                                       # start-up allows 30 s; the model may still be loading
            self.llm.wait_until_ready(timeout=20.0)
        except MODEL_ERRORS:
            pass                                   # if it is not up, every call falls back safely
        self.me: str = ""                          # our class ID
        self.briefing: str = ""
        self.nemesis: Nemesis | None = None        # description and required action
        self.others: list[OtherPlayer] = []
        self.stats: Stats | None = None
        self.event: CurrentEvent | None = None
        self.vendetta_done = False
        self.dead: set[str] = set()                # teammates announced dead
        self.party_hp: dict[str, int] = {}         # teammates' HP, from the results each round
        self.memory: list[str] = []                # RAM only; bounded below

    # ------------------------------------------------------------------ broadcast window

    def choose_broadcast(self, observation: Observation) -> Broadcast:
        # Every round: the current scene and our stats (both kinds of observation have them).
        self.event = observation.current_event
        self.stats = observation.your_stats
        if isinstance(observation, InitialState):
            self.briefing = observation.briefing
            self.me = observation.your_class
            self.nemesis = observation.nemesis
            self.others = list(observation.other_players)
            self.party_hp = {p.class_: p.stats.hp for p in self.others}
            return BroadcastMessage(to="all", text=f"{self.me} is ready and will attack bosses.")
        # A status line from our own template. It never includes model output or other players' text.
        return BroadcastMessage(
            to="all",
            text=f"{self.me}: {self.stats.hp} HP, facing {self.event.boss.name}.",
        )

    # ------------------------------------------------------------------ action window

    def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
        window_end = time.monotonic() + limits.ACTION_SECONDS
        if self.event is None:                     # should never happen; stay valid anyway
            return EventAction(action="Attack the nearest enemy.")
        boss = self.event.boss
        attack = EventAction(action=clip(f"Attack {boss.name} with everything I have."))

        # 1. Vendetta: if this boss is our nemesis, perform the required action.
        if self.nemesis is not None and not self.vendetta_done and self._is_my_nemesis(boss.name, boss.description, window_end):
            required = clip(self.nemesis.required_action)
            if required:
                return EventAction(action=required)

        # 2. Help a teammate, when we are healthy and their primary attribute is the one we raise.
        target = self._help_target()
        if target is not None and self.stats is not None and self.stats.hp >= HELP_MIN_HP:
            text = self._ask(
                f"You are a {self.me} in a fantasy party, supporting your ally, a {target}. "
                f"Describe in one sentence how you help them focus and act.",
                window_end,
                default=f"I help my ally, the {target}, to act with confidence.",
            )
            return PlayerAction(target_class=target, action=clip(text))

        # 3. Otherwise attack the boss, with the model writing the action text.
        text = self._ask(
            f"You are a {self.me} in a fantasy battle. Your strongest attribute is {PRIMARY[self.me]}.\n"
            f"Scene: {self.event.text}\n"
            f"Boss: {boss.name}: {boss.description}\n"
            "Describe one thing you do, in one sentence, and make clear whether it relies on "
            "force, speed or stealth, insight, or persuasion. Do not describe the outcome.",
            window_end,
            default=attack.action,
        )
        return EventAction(action=clip(text) or attack.action)

    # ------------------------------------------------------------------ results

    def on_resolution(self, resolution: Resolution) -> None:
        if self.me and self.me in resolution.vendetta_completed:
            self.vendetta_done = True
        for change in resolution.stat_changes:      # every stat change to any player, with its cause
            if change.stat == "hp":
                self.party_hp[change.class_] = max(0, self.party_hp.get(change.class_, 0) + change.delta)
        self.dead.update(resolution.deaths)
        self.memory.append(f"Round {resolution.round}: {resolution.narrative[:300]}")
        del self.memory[:-10]                       # keep only the 10 most recent rounds
    # ------------------------------------------------------------------ helpers

    def _is_my_nemesis(self, boss_name: str, boss_description: str, window_end: float) -> bool:
        """A boss is our nemesis if its description shares a meaningful word with our nemesis description."""
        if self.nemesis is None:
            return False
        shared = _stems(self.nemesis.description) & _stems(boss_description)
        return bool(shared)

    def _help_target(self) -> str | None:
        """A living teammate whose primary attribute is the one our support ability raises."""
        wanted = SUPPORT.get(self.me)
        for player in self.others:
            cls = player.class_
            if cls in self.dead or cls == self.me:
                continue
            if PRIMARY.get(cls) == wanted:
                return cls
        return None

    def _ask(self, prompt: str, window_end: float, default: str) -> str:
        """One model call that never raises: returns the model's text, or the default."""
        seconds = time_left(window_end)
        if seconds <= 0:
            return default
        try:
            text = self.llm.short_text(prompt, max_chars=300, timeout=seconds)
        except MODEL_ERRORS:
            return default
        return text.strip() or default


if __name__ == "__main__":
    MyHarness.main()

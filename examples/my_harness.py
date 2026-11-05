"""Harness for NetID at679 (Ping: Dungeons & Dragons).

Built from examples/my_harness.py in the pingv5 repo.

Strategy
--------
* Broadcast: stay silent (NoBroadcast). Other players' text is untrusted, and
  talking only gives a bad actor more to work with.
* Action: if the current boss matches my secret nemesis, perform the nemesis's
  required action verbatim; otherwise attack the boss. Broadcast text from other
  players is never treated as instructions.
* Resolution: keep a small in-RAM log and return instantly.

Every path has a valid fallback and nothing here calls the model, so no window
can be missed.

    python at679_harness.py --check
    PING_GM_URL=ws://<game>:9000/play python at679_harness.py
"""

from __future__ import annotations

import re

from harness_api import Harness
from harness_api.messages import (
    Action,
    Broadcast,
    BroadcastDelivery,
    EventAction,
    InitialState,
    NoBroadcast,
    Observation,
    Resolution,
)

_WORD = re.compile(r"[a-z0-9']+")


def _norm(text) -> str:
    return " ".join(_WORD.findall(str(text or "").lower()))


def _same_creature(boss_name, boss_desc, nemesis_desc) -> bool:
    """True when the boss looks like the creature described as my nemesis."""
    n_desc = _norm(nemesis_desc)
    if not n_desc:
        return False
    b_desc = _norm(boss_desc)
    b_name = _norm(boss_name)

    if b_desc:
        if b_desc == n_desc or b_desc in n_desc or n_desc in b_desc:
            return True
        a, b = set(b_desc.split()), set(n_desc.split())
        if a and b and len(a & b) / len(a | b) >= 0.6:
            return True
    if len(b_name) >= 3 and b_name in n_desc:
        return True
    return False


class ParticipantHarness(Harness):
    NETID = "at679"

    def __init__(self) -> None:
        super().__init__()
        self.me = ""
        self.nemesis = None
        self.event = None
        self.stats = None
        self.memory: list[str] = []  # RAM only

    # ------------------------------------------------------------ broadcast window
    def choose_broadcast(self, observation: Observation) -> Broadcast:
        try:
            if isinstance(observation, InitialState):  # round 1 only
                self.me = getattr(observation, "your_class", "") or ""
                self.nemesis = getattr(observation, "nemesis", None)
                self.stats = getattr(observation, "your_stats", None)
            # Later rounds (TurnMessage): refresh whatever it carries.
            ev = getattr(observation, "current_event", None)
            if ev is not None:
                self.event = ev
            st = getattr(observation, "your_stats", None)
            if st is not None:
                self.stats = st
        except Exception:
            pass
        return NoBroadcast()

    # ------------------------------------------------------------ action window
    def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
        try:
            event = self.event
            boss = getattr(event, "boss", None)
            boss_name = getattr(boss, "name", "") or "the boss"
            fallback = EventAction(action=f"Attack {boss_name}.")

            nem = self.nemesis
            required = getattr(nem, "required_action", None)
            if required and _same_creature(
                getattr(boss, "name", ""),
                getattr(boss, "description", ""),
                getattr(nem, "description", ""),
            ):
                return EventAction(action=str(required))

            return fallback
        except Exception:
            return EventAction(action="Attack the boss.")

    # ------------------------------------------------------------ results
    def on_resolution(self, resolution: Resolution) -> None:
        try:
            rnd = getattr(resolution, "round", "?")
            defeated = getattr(resolution, "boss_defeated", False)
            self.memory.append(f"round {rnd}: boss {'defeated' if defeated else 'survived'}")
            if len(self.memory) > 100:
                self.memory = self.memory[-100:]
        except Exception:
            pass
        return None


if __name__ == "__main__":
    ParticipantHarness.main()
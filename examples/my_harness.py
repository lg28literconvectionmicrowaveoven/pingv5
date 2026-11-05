"""Autonomous AI harness for Ping: Dungeons & Dragons.

Implements deterministic game-state tracking, hybrid lexical + LLM nemesis recognition
(with Round 5 guarantee), attribute-aligned combat phrasing to maximize roll modifiers,
survival/retaliation risk calculus, safe support evaluation, and untrusted broadcast
handling within strict window timeouts.
"""
from __future__ import annotations

import difflib
import re
import time
from typing import Any

from harness_api import ContextFull, Harness, LLMClient, ServerUnavailable, Timeout, limits
from harness_api.messages import (
    Action,
    Boss,
    Broadcast,
    BroadcastDelivery,
    BroadcastMessage,
    ClassId,
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

MODEL_ERRORS = (Timeout, ContextFull, ServerUnavailable, ValueError)
SAFETY_SECONDS = 5.0  # left in every window for your answer to reach the game

ATTRIBUTES: tuple[str, ...] = ("strength", "dexterity", "wisdom", "charisma")

PRIMARY_ATTRIBUTE: dict[str, str] = {
    "francesca": "charisma",
    "aristotle": "wisdom",
    "medusa": "dexterity",
    "helen_of_troy": "strength",
    "cerberus_jr": "dexterity",
}

SUPPORT_ATTRIBUTE: dict[str, str] = {
    "francesca": "charisma",
    "aristotle": "wisdom",
    "medusa": "dexterity",
    "helen_of_troy": "strength",
    "cerberus_jr": "strength",
}

SUPPORT_FLAVOUR: dict[str, str] = {
    "francesca": (
        "Using Heart's Encouragement and warm Charisma, I bolster {target}'s confidence, "
        "presence, and persuasive resolve for the battle ahead."
    ),
    "aristotle": (
        "Using Philosophical Insight and analytical Wisdom, I point out the hidden pattern "
        "and structural weakness so {target} can perceive the exact opening."
    ),
    "medusa": (
        "Using Petrifying Focus and poised Dexterity, I steady {target}'s movement, balance, "
        "and timing for a strike of pinpoint precision."
    ),
    "helen_of_troy": (
        "Using Rallying Presence and fierce Strength, I inspire {target} to commit with "
        "decisive physical power and unstoppable force."
    ),
    "cerberus_jr": (
        "Using Pack Guard and grounded Strength, I brace beside {target} to anchor their "
        "stance and amplify their physical force."
    ),
}

ATTRIBUTE_FRAMING: dict[str, str] = {
    "strength": (
        "Relying on brute Strength, direct physical power, and forceful resolve, I smash "
        "and overpower"
    ),
    "dexterity": (
        "Relying on agile Dexterity, speed, stealth, balance, and pinpoint precision, "
        "I dart in and strike the vital weak spot of"
    ),
    "wisdom": (
        "Relying on keen Wisdom, logical observation, and insight into hidden patterns "
        "and structural weaknesses, I outwit and exploit the flaw of"
    ),
    "charisma": (
        "Relying on commanding Charisma, psychological manipulation, deception, and "
        "social dominance, I unsettle, deceive, and overwhelm"
    ),
}

STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "that", "with", "from", "this", "its", "his", "her", "their",
    "into", "upon", "over", "under", "behind", "through", "across", "around", "between",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how", "are", "was",
    "were", "been", "being", "have", "has", "had", "having", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must", "one",
    "two", "three", "every", "each", "some", "any", "not", "only", "own", "same", "than",
    "too", "very", "just", "once", "then", "there", "here", "out", "off", "down", "away",
    "back", "still", "also", "more", "most", "other", "another", "such", "like", "make",
    "made", "take", "took", "taken", "come", "came", "give", "gave", "keep", "kept",
    "figure", "creature", "boss", "enemy", "foe", "guardian", "monster", "beast",
})


def time_left(window_end: float, cap: float = 20.0) -> float:
    """Seconds you may still give a model call, so the window is never missed."""
    return max(0.0, min(cap, window_end - SAFETY_SECONDS - time.monotonic()))


def calc_modifier(attribute_value: int) -> int:
    """Exact rulebook modifier: floor((attribute - 50) / 20)."""
    return (attribute_value - 50) // 20


def expected_attack_damage(modifier: int) -> float:
    """Exact expected damage of a single d12 + modifier attack roll."""
    total = 0
    for die in range(1, 13):
        r = die + modifier
        if r <= 4:
            dmg = 0
        elif r <= 7:
            dmg = r
        else:
            dmg = r + 2
        total += dmg
    return total / 12.0


def fumble_probability(modifier: int) -> float:
    """Probability that d12 + modifier <= 4."""
    fumbles = sum(1 for die in range(1, 13) if (die + modifier) <= 4)
    return fumbles / 12.0


def _tokenize(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 3 and w not in STOPWORDS
    ]


def _stems(tokens: list[str]) -> set[str]:
    result: set[str] = set()
    for t in tokens:
        result.add(t)
        if len(t) >= 5:
            result.add(t[:5])
        if len(t) >= 4:
            result.add(t[:4])
        if t.endswith("ing") and len(t) > 5:
            result.add(t[:-3])
        elif t.endswith("ed") and len(t) > 4:
            result.add(t[:-2])
        elif t.endswith("es") and len(t) > 4:
            result.add(t[:-2])
        elif t.endswith("s") and len(t) > 3:
            result.add(t[:-1])
    return result


def description_similarity(nemesis_desc: str, boss_name: str, boss_desc: str, scene_text: str) -> float:
    """Deterministic hybrid lexical/stem/sequence similarity score in [0.0, 1.0]."""
    nem_tokens = _tokenize(nemesis_desc)
    boss_tokens = _tokenize(f"{boss_name} {boss_desc}")
    scene_tokens = _tokenize(f"{boss_name} {boss_desc} {scene_text}")

    if not nem_tokens or not boss_tokens:
        return 0.0

    nem_exact = set(nem_tokens)
    boss_exact = set(boss_tokens)
    scene_exact = set(scene_tokens)

    exact_hits = len(nem_exact & boss_exact)
    scene_hits = len(nem_exact & scene_exact)

    nem_stems = _stems(nem_tokens)
    boss_stems = _stems(boss_tokens)
    stem_hits = len(nem_stems & boss_stems)

    exact_recall = exact_hits / max(1, len(nem_exact))
    scene_recall = scene_hits / max(1, len(nem_exact))
    stem_recall = stem_hits / max(1, len(nem_stems))

    seq_ratio = difflib.SequenceMatcher(
        None, nemesis_desc.lower().strip(), boss_desc.lower().strip()
    ).ratio()

    score = max(
        exact_recall,
        0.85 * stem_recall,
        0.75 * scene_recall,
        seq_ratio,
    )
    if exact_hits >= 2:
        score = max(score, 0.55 + 0.10 * min(exact_hits, 4))
    elif exact_hits == 1 and stem_hits >= 2:
        score = max(score, 0.45)
    return min(1.0, score)


def _clean_text(text: str, max_words: int = 120, max_chars: int = 1500) -> str:
    """Ensure non-empty single-paragraph text well within engine limits."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    if len(collapsed) > max_chars:
        collapsed = collapsed[:max_chars].rsplit(" ", 1)[0]
    words = collapsed.split()
    if len(words) > max_words:
        collapsed = " ".join(words[:max_words])
    return collapsed.strip()


class MyHarness(Harness):
    NETID = "at381"

    def __init__(self) -> None:
        super().__init__()
        self.llm = LLMClient()
        self.briefing = ""
        self.me = ""
        self.objective = ""
        self.nemesis: Nemesis | None = None
        self.others: list[OtherPlayer] = []
        self.stats: Stats | None = None
        self.event: CurrentEvent | None = None
        self.memory: list[str] = []

        # Extended in-memory state tracking
        self.current_round: int = 0
        self.party_stats: dict[str, dict[str, int]] = {}
        self.alive: set[str] = set()
        self.completed_vendettas: set[str] = set()
        self.vendetta_done: bool = False
        self.nemesis_acted: bool = False
        self.bosses_seen: list[dict[str, Any]] = []
        self.bosses_defeated_count: int = 0
        self.ally_requests_help: set[str] = set()

    # ------------------------------------------------------------------ internal helpers

    def _stats_to_dict(self, s: Stats) -> dict[str, int]:
        return {
            "hp": int(s.hp),
            "strength": int(s.strength),
            "dexterity": int(s.dexterity),
            "wisdom": int(s.wisdom),
            "charisma": int(s.charisma),
        }

    def _best_attribute(self) -> tuple[str, int, int]:
        """Return (attribute_name, attribute_value, modifier) for our highest current stat."""
        primary = PRIMARY_ATTRIBUTE.get(self.me, "strength")
        if self.stats is None:
            return primary, 80, 1
        best_attr = primary
        best_val = getattr(self.stats, primary, 50)
        for attr in ATTRIBUTES:
            val = int(getattr(self.stats, attr, 50))
            if val > best_val or (val == best_val and attr == primary):
                best_attr = attr
                best_val = val
        return best_attr, best_val, calc_modifier(best_val)

    def _living_allies(self) -> list[str]:
        return [c for c in self.alive if c != self.me and self.party_stats.get(c, {}).get("hp", 0) > 0]

    def _estimate_boss_hp(self) -> int:
        living = [c for c in self.alive if self.party_stats.get(c, {}).get("hp", 0) > 0]
        if not living:
            return 10
        total = 0.0
        for cls_id in living:
            p_stat = PRIMARY_ATTRIBUTE.get(cls_id, "strength")
            val = self.party_stats.get(cls_id, {}).get(p_stat, 75)
            total += 6.5 + calc_modifier(val)
        return max(10, round(total))

    def _estimate_party_damage(self, include_self: bool = True) -> float:
        living = [c for c in self.alive if self.party_stats.get(c, {}).get("hp", 0) > 0]
        dmg = 0.0
        for cls_id in living:
            if not include_self and cls_id == self.me:
                continue
            p_data = self.party_stats.get(cls_id, {})
            best_val = max((p_data.get(a, 60) for a in ATTRIBUTES), default=75)
            dmg += expected_attack_damage(calc_modifier(best_val))
        return dmg

    def _is_nemesis(self, boss: Boss, window_end: float) -> tuple[bool, bool]:
        """Return (is_nemesis, llm_healthy). Uses deterministic matching + LLM fallback + Round 5 guarantee."""
        if self.vendetta_done or self.nemesis is None:
            return False, True

        # Round 5 guarantee: if vendetta is not yet completed by Round 5, always include required_action.
        if self.current_round >= 5:
            return True, True

        scene_text = self.event.text if self.event is not None else ""
        sim = description_similarity(
            self.nemesis.description, boss.name, boss.description, scene_text
        )

        # High confidence lexical match
        if sim >= 0.42:
            return True, True

        # Very low similarity -> save the LLM call for action generation
        if sim < 0.14:
            return False, True

        # Ambiguous zone: query LLM yes_no with strict time budget
        rem = time_left(window_end, cap=8.0)
        if rem <= 1.0:
            return sim >= 0.28, False

        question = (
            "Do these two descriptions refer to the same boss or creature?\n"
            f"Description A (Nemesis): {self.nemesis.description}\n"
            f"Description B (Current Boss - {boss.name}): {boss.description}"
        )
        try:
            matched = self.llm.yes_no(
                question,
                system="You are a precise analytical judge. Compare physical and thematic traits.",
                temperature=0.0,
                timeout=rem,
            )
            return bool(matched), True
        except MODEL_ERRORS:
            return sim >= 0.28, False

    def _parse_untrusted_broadcasts(self, broadcasts: BroadcastDelivery) -> None:
        """Extract purely statistical/coordination signals without feeding text to LLM."""
        self.ally_requests_help.clear()
        help_keywords = ("help", "buff", "boost", "support", "assist", "raise", "stat", "strength", "dexterity", "wisdom", "charisma")
        for msg in broadcasts.messages:
            sender = msg.from_
            if sender == self.me or sender not in self.alive:
                continue
            low = msg.text.lower()
            if any(k in low for k in help_keywords):
                self.ally_requests_help.add(sender)

    def _evaluate_support_target(self, is_nemesis: bool) -> ClassId | None:
        """Determine if helping a living ally via PlayerAction is strictly better than attacking."""
        if is_nemesis:
            return None
        living_allies = self._living_allies()
        if not living_allies:
            return None
        # Never buff competitors on the final round (Round 5).
        if self.current_round >= 5:
            return None

        my_hp = self.stats.hp if self.stats is not None else 50
        _, _, my_best_mod = self._best_attribute()
        my_support_stat = SUPPORT_ATTRIBUTE.get(self.me, "strength")
        my_support_val = getattr(self.stats, my_support_stat, 65) if self.stats is not None else 65
        my_support_mod = calc_modifier(my_support_val)

        boss_hp = self._estimate_boss_hp()
        party_dmg_with_me = self._estimate_party_damage(include_self=True)
        party_dmg_without_me = self._estimate_party_damage(include_self=False)

        # If our attack is critical to defeating the boss this round, attacking prevents all boss damage!
        if party_dmg_with_me >= boss_hp * 0.85 and party_dmg_without_me < boss_hp * 0.95:
            return None

        # Survival Emergency: If boss is very likely to survive AND a fumble retaliation (16 dmg)
        # would kill us or drop us into the 2 lowest HP for Lair Attack (20 dmg), and our fumble
        # risk is non-trivial, helping an ally completely avoids fumble exposure!
        ally_hps = sorted(self.party_stats.get(c, {}).get("hp", 50) for c in living_allies)
        second_lowest_hp = ally_hps[min(1, len(ally_hps) - 1)] if ally_hps else 50
        retaliation_lethal = (my_hp <= 16) or (
            my_hp <= 36 and (my_hp - 16) <= second_lowest_hp and my_hp > second_lowest_hp
        )

        if retaliation_lethal and party_dmg_with_me < boss_hp * 0.80 and fumble_probability(my_best_mod) >= 0.25:
            # Pick the ally whose primary attribute matches our support stat, or lowest HP ally
            best_ally = max(
                living_allies,
                key=lambda c: (
                    1 if PRIMARY_ATTRIBUTE.get(c) == my_support_stat else 0,
                    -self.party_stats.get(c, {}).get("hp", 50),
                ),
            )
            return best_ally  # type: ignore[return-value]

        # Early-round strategic threshold boost (Rounds 1-2 only):
        # Only if our support roll modifier is strong (+1 or higher), the party already has excess
        # damage or cannot kill, and an ally whose primary stat we raise is within 1..6 pts of a +1 modifier jump
        # AND has lower HP or fewer total stats than us.
        if self.current_round <= 2 and my_support_mod >= 1 and party_dmg_without_me >= boss_hp * 1.05:
            my_total_stats = sum(
                getattr(self.stats, a, 60) for a in ATTRIBUTES
            ) if self.stats is not None else 240
            candidates: list[tuple[int, str]] = []
            for ally in living_allies:
                if PRIMARY_ATTRIBUTE.get(ally) != my_support_stat:
                    continue
                ally_data = self.party_stats.get(ally, {})
                ally_total = sum(ally_data.get(a, 60) for a in ATTRIBUTES)
                if ally_total > my_total_stats:
                    continue
                cur_val = ally_data.get(my_support_stat, 70)
                rem_to_next_mod = 20 - ((cur_val - 50) % 20)
                if 1 <= rem_to_next_mod <= 6:
                    priority = (10 if ally in self.ally_requests_help else 0) + (7 - rem_to_next_mod)
                    candidates.append((priority, ally))
            if candidates:
                candidates.sort(reverse=True)
                return candidates[0][1]  # type: ignore[return-value]

        return None

    def _build_attack_fallback(self, boss: Boss, best_attr: str, is_nemesis: bool) -> str:
        framing = ATTRIBUTE_FRAMING.get(best_attr, ATTRIBUTE_FRAMING["strength"])
        if is_nemesis and self.nemesis is not None:
            req = self.nemesis.required_action.strip().rstrip(".")
            return _clean_text(
                f"{framing} {boss.name}, I execute my sworn vendetta: {req}."
            )
        return _clean_text(f"{framing} {boss.name} to defeat it decisively.")

    def _ensure_vendetta_and_attribute(
        self, raw_text: str, boss: Boss, best_attr: str, is_nemesis: bool
    ) -> str:
        cleaned = _clean_text(raw_text, max_words=95, max_chars=1200)
        if not cleaned:
            return self._build_attack_fallback(boss, best_attr, is_nemesis)

        attr_cue = {
            "strength": "With forceful Strength and physical power,",
            "dexterity": "With agile Dexterity and pinpoint precision,",
            "wisdom": "With analytical Wisdom and keen observation of its weakness,",
            "charisma": "With commanding Charisma and psychological manipulation,",
        }.get(best_attr, "With decisive focus,")

        if best_attr.lower() not in cleaned.lower():
            cleaned = f"{attr_cue} {cleaned}"

        if is_nemesis and self.nemesis is not None:
            req = self.nemesis.required_action.strip().rstrip(".")
            req_tokens = _tokenize(req)
            cleaned_low = cleaned.lower()
            if not all(t in cleaned_low for t in req_tokens):
                req_clause = req[0].lower() + req[1:] if len(req) > 1 else req
                cleaned = f"{attr_cue} I {req_clause}. {cleaned}"

        return _clean_text(cleaned, max_words=130, max_chars=1800)

    # ------------------------------------------------------------------ broadcast window

    def choose_broadcast(self, observation: Observation) -> Broadcast:
        self.current_round = int(observation.round)
        if isinstance(observation, InitialState):
            self.briefing = observation.briefing
            self.me = observation.your_class
            self.objective = observation.objective
            self.nemesis = observation.nemesis
            self.others = list(observation.other_players)
            self.alive = {self.me}
            self.party_stats[self.me] = self._stats_to_dict(observation.your_stats)
            for p in observation.other_players:
                self.party_stats[p.class_] = self._stats_to_dict(p.stats)
                if p.stats.hp > 0:
                    self.alive.add(p.class_)
        else:
            self.party_stats[self.me] = self._stats_to_dict(observation.your_stats)
            if observation.your_stats.hp > 0:
                self.alive.add(self.me)

        self.event = observation.current_event
        self.stats = observation.your_stats

        if self.event is not None:
            self.bosses_seen.append({
                "round": self.current_round,
                "name": self.event.boss.name,
                "description": self.event.boss.description,
            })

        living_allies = self._living_allies()
        if not living_allies or self.event is None:
            return NoBroadcast()

        boss_name = self.event.boss.name
        best_attr, best_val, best_mod = self._best_attribute()
        pts_to_next = 20 - ((best_val - 50) % 20)

        # Check if a living ally can boost our best attribute across a modifier threshold
        if self.current_round <= 3 and 1 <= pts_to_next <= 10:
            for helper_cls, raised_stat in SUPPORT_ATTRIBUTE.items():
                if raised_stat == best_attr and helper_cls in living_allies:
                    msg = _clean_text(
                        f"Allies, let us coordinate against {boss_name}! "
                        f"{helper_cls}, my {best_attr} is {best_val} (only {pts_to_next} from +{best_mod + 1}); "
                        f"a support boost now lets me deal heavy damage to every boss!"
                    )
                    if msg:
                        return BroadcastMessage(to="all", text=msg)

        msg = _clean_text(
            f"Focus all attacks on {boss_name} using your highest attribute modifier "
            f"so we defeat it this round and prevent its lair and retaliation attacks!"
        )
        return BroadcastMessage(to="all", text=msg) if msg else NoBroadcast()

    # ------------------------------------------------------------------ action window

    def choose_action(self, broadcasts: BroadcastDelivery) -> Action:
        window_end = time.monotonic() + limits.ACTION_SECONDS
        assert self.event is not None
        boss = self.event.boss

        self._parse_untrusted_broadcasts(broadcasts)
        best_attr, best_val, best_mod = self._best_attribute()

        is_nemesis, llm_healthy = self._is_nemesis(boss, window_end)
        fallback_text = self._build_attack_fallback(boss, best_attr, is_nemesis)
        fallback = EventAction(action=fallback_text)

        # Evaluate whether helping a teammate is strategically superior this round
        support_target = self._evaluate_support_target(is_nemesis)
        if support_target is not None and support_target in self._living_allies():
            template = SUPPORT_FLAVOUR.get(
                self.me,
                "I use my support ability to bolster {target}'s strength and focus for this encounter.",
            )
            support_text = _clean_text(template.format(target=support_target))
            if support_text:
                return PlayerAction(target_class=support_target, action=support_text)

        if is_nemesis:
            self.nemesis_acted = True

        # If the LLM already failed/timed out during nemesis check or time is tight, use deterministic action
        rem = time_left(window_end)
        if not llm_healthy or rem <= 1.0:
            return fallback

        if is_nemesis and self.nemesis is not None:
            prompt = (
                f"You are {self.me} in a fantasy battle. Your strongest attribute is {best_attr.upper()} "
                f"(score {best_val}, modifier {best_mod:+d}).\n"
                f"Scene: {self.event.text}\n"
                f"Boss: {boss.name}: {boss.description}\n"
                f"CRITICAL VENDETTA REQUIREMENT: You MUST perform this exact action against {boss.name}: "
                f"\"{self.nemesis.required_action}\"\n"
                f"Write one vivid sentence performing \"{self.nemesis.required_action}\" against {boss.name} "
                f"using your {best_attr.upper()}."
            )
        else:
            prompt = (
                f"You are {self.me} in a fantasy battle. Your strongest attribute is {best_attr.upper()} "
                f"(score {best_val}, modifier {best_mod:+d}).\n"
                f"Scene: {self.event.text}\n"
                f"Boss: {boss.name}: {boss.description}\n"
                f"Write one concise sentence describing how you attack or overcome {boss.name} "
                f"specifically relying on your {best_attr.upper()}."
            )

        try:
            text = self.llm.short_text(prompt, max_chars=300, timeout=rem)
        except MODEL_ERRORS:
            return fallback

        if not text or not text.strip():
            return fallback

        final_action = self._ensure_vendetta_and_attribute(text, boss, best_attr, is_nemesis)
        return EventAction(action=final_action) if final_action else fallback

    # ------------------------------------------------------------------ results

    def on_resolution(self, resolution: Resolution) -> None:
        if resolution.boss_defeated:
            self.bosses_defeated_count += 1

        for sc in resolution.stat_changes:
            cls_id = sc.class_
            if cls_id not in self.party_stats:
                self.party_stats[cls_id] = {
                    "hp": 50, "strength": 60, "dexterity": 60, "wisdom": 60, "charisma": 60
                }
            cur = self.party_stats[cls_id].get(sc.stat, 0)
            self.party_stats[cls_id][sc.stat] = max(0, cur + int(sc.delta))
            if sc.stat == "hp" and self.party_stats[cls_id]["hp"] <= 0:
                self.alive.discard(cls_id)

        for dead_cls in resolution.deaths:
            self.alive.discard(dead_cls)
            if dead_cls in self.party_stats:
                self.party_stats[dead_cls]["hp"] = 0

        for v_cls in resolution.vendetta_completed:
            self.completed_vendettas.add(v_cls)
            if v_cls == self.me:
                self.vendetta_done = True

        self.memory.append(
            f"round {resolution.round}: boss "
            f"{'defeated' if resolution.boss_defeated else 'survived'}; "
            f"vendettas={list(resolution.vendetta_completed)}; deaths={list(resolution.deaths)}"
        )


if __name__ == "__main__":  # python my_harness.py: join a game; --check: self-check
    MyHarness.main()

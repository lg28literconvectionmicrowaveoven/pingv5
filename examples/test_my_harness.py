from __future__ import annotations

from harness_api import ContextFull, Harness, LLMClient, ServerUnavailable, Timeout, limits
from harness_api.messages import (
    Action,
    Broadcast,
    EventAction,
    NoBroadcast,
    Observation,
    BroadcastDelivery,
)

class MyHarness(Harness):
    NETID = "at679"

    def __init__(self):
        super().__init__()
        self.me = None
        self.nemesis = None

    def choose_broadcast(self, observation: Observation) -> Broadcast:
        if observation.type == "initial_state":
            self.me = observation.your_class
            self.nemesis = observation.nemesis
        return NoBroadcast(to="none")

    def choose_action(self, delivery: BroadcastDelivery) -> Action:
        # Fallback action if model fails or times out
        fallback = EventAction(target="event", action="Attack the boss cautiously.")
        
        try:
            prompt = f"""You are {self.me} in a strategic fantasy battle. 
Your nemesis requires action: {getattr(self.nemesis, 'required_action', 'None')}
Choose your next action carefully."""
            
            response = LLMClient().short_text(
                prompt,
                system="You are a strategic fantasy game player.",
                timeout=limits.ACTION_SECONDS - 5
            )
            if not response or not response.strip():
                return fallback
            return EventAction(target="event", action=response.strip())
        except (Timeout, ServerUnavailable, ContextFull, ValueError, Exception):
            return fallback

    def on_resolution(self, resolution) -> None:
        pass
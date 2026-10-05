"""Pure cross-node reconciliation of knob wants (judge D1d, STORMER-LOOP-DESIGN sec 4.10).  No MPI/torch imports.

Legacy behaviour (STORMER_CONSENSUS unset/"max") is an element-wise MPI MAX over all ranks: an increase requested on one
node beats a hold or revert on another, and the outvoted node's planner then labels samples with an arm that was never
applied.  Unanimous mode replaces it:

  * node-local knobs (num_workers, prefetch_factor, accumulate_grad_batches, decode_threads): MAX *within the node*
    only.  With per-node context scope one rank drains the node's plan, so this is a broadcast to that node's ranks;
    nodes may hold different values and each planner's ack reports its own observed state.
  * bucket_cap_mb (a WORLD collective, DDP rebuild): applied only when EVERY node votes and all votes are equal
    (unanimity-or-hold).  A node's vote is sticky for `sticky` windows so plans that reach nodes a window apart can
    still meet; an expired or superseded vote is acknowledged `failed` so its planner reverts to the applied value.
Every window with a vote records an agreement outcome; `agreement_rate` is unanimous / voting windows.
"""
from __future__ import annotations

SENT = -1
BIG = 2 ** 62


class Sticky:
    """One rank's pending bucket_cap vote (node leaders only; others never vote)."""

    def __init__(self, sticky_windows: int = 3):
        self.limit = max(int(sticky_windows), 1)
        self.value = SENT
        self.age = 0
        self.actions = []          # (action, knob, old) awaiting a verdict

    def offer(self, value: int, actions):
        """New local vote this window.  Returns actions superseded by a different value (to be acked failed)."""
        dropped = []
        if value > SENT:
            if self.value > SENT and value != self.value:
                dropped, self.actions = self.actions, []
            if value != self.value:
                self.age = 0
            self.value = value
            self.actions = list(self.actions) + list(actions)
        elif self.value > SENT:
            self.age += 1
        return dropped

    def clear(self):
        self.value, self.age, acts = SENT, 0, self.actions
        self.actions = []
        return acts

    def expired(self) -> bool:
        return self.value > SENT and self.age >= self.limit


def contribution(is_leader: bool, vote: int):
    """(leader, has_vote, min_key, neg_max_key) for world SUM/SUM/MIN/MIN reductions."""
    has = 1 if (is_leader and vote > SENT) else 0
    return (1 if is_leader else 0, has, vote if has else BIG, -vote if has else BIG)


def decide(n_leaders: int, n_votes: int, vmin: int, neg_vmax: int):
    """('none'|'unanimous'|'hold', value).  Same answer on every rank (inputs are the reduced values)."""
    if n_votes == 0:
        return "none", SENT
    vmax = -neg_vmax
    if n_votes == n_leaders and vmin == vmax:
        return "unanimous", vmax
    return "hold", SENT


class Agreement:
    def __init__(self):
        self.voting = 0
        self.unanimous = 0

    def note(self, outcome: str):
        if outcome in ("unanimous", "hold"):
            self.voting += 1
            self.unanimous += outcome == "unanimous"

    @property
    def agreement_rate(self):
        return None if not self.voting else self.unanimous / self.voting

"""Simulated multi-node runs of consensus.py with a fake world reduction (no MPI)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import consensus as c


def world_step(stickies, votes, sticky_ok=True):
    """One window: each node leader offers its vote; the world reduces; returns (outcome, value, dropped, expired)."""
    dropped = [s.offer(v, [("act%d" % i, "bucket_cap_mb", 25)] if v > c.SENT else []) for i, (s, v) in enumerate(zip(stickies, votes))]
    parts = [c.contribution(True, s.value) for s in stickies]
    n_l, n_v = sum(p[0] for p in parts), sum(p[1] for p in parts)
    outcome, value = c.decide(n_l, n_v, min(p[2] for p in parts), min(p[3] for p in parts))
    return outcome, value, dropped


def test_unanimous_applies_and_clears_and_hold_is_not_max():
    st = [c.Sticky(3) for _ in range(3)]
    out, val, _ = world_step(st, [50, 50, 50])
    assert (out, val) == ("unanimous", 50)
    st = [c.Sticky(3) for _ in range(3)]
    out, val, _ = world_step(st, [100, 50, c.SENT])          # legacy MAX would apply 100
    assert (out, val) == ("hold", c.SENT)


def test_votes_a_window_apart_meet_via_sticky_and_expiry_fails_the_vote():
    st = [c.Sticky(3) for _ in range(2)]
    assert world_step(st, [50, c.SENT])[0] == "hold"
    out, val, _ = world_step(st, [c.SENT, 50])               # second node arrives one window later
    assert (out, val) == ("unanimous", 50)
    st = [c.Sticky(2) for _ in range(2)]
    world_step(st, [50, c.SENT]); world_step(st, [c.SENT, c.SENT]); world_step(st, [c.SENT, c.SENT])
    assert st[0].expired() and st[0].clear()[0][1] == "bucket_cap_mb" and st[0].value == c.SENT


def test_superseded_vote_is_returned_for_a_failed_ack_and_agreement_rate():
    s = c.Sticky(3)
    s.offer(50, [("a", "bucket_cap_mb", 25)])
    assert s.offer(100, [("b", "bucket_cap_mb", 25)])[0][0] == "a" and s.value == 100
    a = c.Agreement(); [a.note(o) for o in ("none", "hold", "unanimous", "unanimous")]
    assert a.agreement_rate == 2 / 3

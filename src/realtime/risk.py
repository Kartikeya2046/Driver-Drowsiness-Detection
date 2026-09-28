"""p_now (GRU sigmoid output) -> discrete alert level 0-3 for the Arduino protocol
(R:<level>\\n, see context.md). Thresholds are a starting point for the demo script
(green -> yellow/vibration -> buzzer); tune against the Phase 3 lead-time curve
(results/phase3_curve.csv) once real footage/hardware is available.

recenter() exists because the GRU's raw output level varies a lot by person: a
regression check on known lab labels found one subject scoring 0.816 (already
"red") on their OWN known-alert footage, and other subjects barely separating
alert from drowsy at all - same GRU, same code, just per-person variability the
model was never going to fully calibrate out. The global thresholds above were
never tuned to any individual's own baseline. recenter() reuses the calibration
window itself (already collected for feature calibration) to also measure this
person's own baseline model output, then maps live p_now relative to that -
logit-space, not a plain subtraction, so the result stays a valid [0,1] probability.
"""
import math

LEVELS = ((0.75, 3), (0.5, 2), (0.3, 1))  # (min p_now, level), checked high to low
TARGET_BASELINE = 0.10  # what a person's own calibration-window baseline should map to


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def recenter(p_now: float, baseline_p: float, target: float = TARGET_BASELINE) -> float:
    """Shift p_now so this person's own calibration-window baseline maps to `target`
    instead of wherever the raw model happened to put it."""
    return sigmoid(logit(p_now) - logit(baseline_p) + logit(target))


MICROSLEEP_S = 1.5  # continuous eyes-closed past this -> force at least level 2
LONG_CLOSURE_S = 3.0  # ... past this -> force level 3
# ponytail: fixed cutoffs, not tuned against real closure-duration data (none exists
# yet - no hardware/user sessions to calibrate against); reasonable microsleep-literature
# starting points. Retune once real sessions give a false-alarm/miss rate to optimize.


def apply_closure_override(level: int, closed_s: float) -> int:
    """The GRU reasons over a 45s window, so a few seconds of closed eyes barely
    moves it - sustained closure forces the level up regardless of what the GRU says,
    checked every tick (~100ms) rather than waiting on the windowed model."""
    if closed_s >= LONG_CLOSURE_S:
        return max(level, 3)
    if closed_s >= MICROSLEEP_S:
        return max(level, 2)
    return level


def risk_level(p_now: float) -> int:
    for thr, level in LEVELS:
        if p_now >= thr:
            return level
    return 0


def demo() -> None:
    assert risk_level(0.0) == 0
    assert risk_level(0.29) == 0
    assert risk_level(0.3) == 1
    assert risk_level(0.49) == 1
    assert risk_level(0.5) == 2
    assert risk_level(0.74) == 2
    assert risk_level(0.75) == 3
    assert risk_level(1.0) == 3
    # recenter: at p_now == baseline, output == target (the whole point)
    assert abs(recenter(0.816, 0.816, target=0.10) - 0.10) < 1e-6
    # a subject whose own alert baseline was 0.816 (real regression-check case):
    # their genuinely-drowsy 0.994 should still land high after recentering
    assert recenter(0.994, 0.816, target=0.10) > 0.5
    # a well-behaved subject (baseline 0.05) barely above their own baseline stays low
    assert recenter(0.10, 0.05, target=0.10) < 0.3
    # closure override: doesn't kick in early, forces up past each cutoff, never lowers
    assert apply_closure_override(0, 1.0) == 0
    assert apply_closure_override(0, 1.5) == 2
    assert apply_closure_override(1, 2.0) == 2
    assert apply_closure_override(0, 3.0) == 3
    assert apply_closure_override(3, 0.0) == 3  # never overrides downward
    print("risk.py: ok")


if __name__ == "__main__":
    demo()

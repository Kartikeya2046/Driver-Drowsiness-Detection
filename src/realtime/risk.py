"""p_now (GRU sigmoid output) -> discrete alert level 0-3 for the Arduino protocol
(R:<level>\\n, see context.md). Thresholds are a starting point for the demo script
(green -> yellow/vibration -> buzzer); tune against the Phase 3 lead-time curve
(results/phase3_curve.csv) once real footage/hardware is available.
"""
LEVELS = ((0.75, 3), (0.5, 2), (0.3, 1))  # (min p_now, level), checked high to low


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
    print("risk.py: ok")


if __name__ == "__main__":
    demo()

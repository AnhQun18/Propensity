"""Live progress tracking and strict <30m per-model runtime enforcement."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


def fmt_sec(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60.0:
        return f"{sec:.1f}s"
    mins, rem = divmod(int(sec), 60)
    if mins < 60:
        return f"{mins}m{rem:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h{mins:02d}m{rem:02d}s"


@dataclass
class ModelProgressTracker:
    """Prints real-time step/fold progress and ETA while enforcing max model runtime."""

    stage_name: str
    exp_id: str
    total_units: int
    max_seconds: float = 1800.0
    t_start: float = field(default_factory=time.perf_counter)
    completed_units: int = 0

    def step_done(self, label: str, extra: str = "") -> float:
        self.completed_units += 1
        elapsed = time.perf_counter() - self.t_start
        rate = elapsed / max(1, self.completed_units)
        eta = rate * max(0, self.total_units - self.completed_units)
        pct = 100.0 * self.completed_units / max(1, self.total_units)
        msg = (
            f"  [{self.stage_name} | {self.exp_id}] "
            f"{self.completed_units}/{self.total_units} ({pct:5.1f}%) "
            f"{label} | elapsed={fmt_sec(elapsed)} | ETA={fmt_sec(eta)}"
        )
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)
        if elapsed > self.max_seconds:
            raise TimeoutError(
                f"Model {self.exp_id} exceeded max_seconds={self.max_seconds:.0f}s "
                f"(elapsed={elapsed:.1f}s)"
            )
        return elapsed

    def finish(self, summary: str = "") -> float:
        elapsed = time.perf_counter() - self.t_start
        msg = f"[DONE {self.stage_name} | {self.exp_id}] total={fmt_sec(elapsed)}"
        if summary:
            msg += f" | {summary}"
        print(msg, flush=True)
        return elapsed

"""Clock shared by deadlines in every process of a run.

On one host the monotonic clock is used (default). When roles run on several
hosts (distributed mode), deadlines travel in message headers and must be
compared on the receiving host, so all processes switch to the wall clock,
which cloud VMs keep synchronized with NTP/chrony (skew of milliseconds,
far below the round deadlines of seconds). Set ADAPTIVE_CLOCK=wall in every
process of a distributed run; the launcher does this automatically.
"""

import os
import time

MODE = os.environ.get("ADAPTIVE_CLOCK", "monotonic")
if MODE not in ("monotonic", "wall"):
    raise ValueError("ADAPTIVE_CLOCK must be 'monotonic' or 'wall'")

now = time.time if MODE == "wall" else time.monotonic

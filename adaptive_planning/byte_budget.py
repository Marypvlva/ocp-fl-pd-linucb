"""A hard application-frame byte limit shared by senders in the local stand."""

from contextlib import contextmanager


class ByteBudgetExceeded(ValueError):
    pass


class SharedByteBudget:
    """Reserve complete frames, account successful sends, release unsent tails.

    The lock guards only counters, never a socket operation. A terminated
    sender can leave a conservative reservation, which prevents overspending.
    Management frames with round zero are excluded by the transport caller.
    """

    def __init__(self, context, limit):
        if type(limit) is not int or not 0 < limit < 2**63:
            raise ValueError("The application byte limit must be a positive int64")
        self.limit = limit
        self.spent = context.Value("q", 0, lock=False)
        self.reserved = context.Value("q", 0, lock=False)
        self.denied = context.Value("q", 0, lock=False)
        self.lock = context.Lock()

    @contextmanager
    def guard(self):
        if not self.lock.acquire(timeout=1.0):
            raise TimeoutError("Application byte ledger lock unavailable")
        try:
            yield
        finally:
            self.lock.release()

    def reserve(self, size):
        if type(size) is not int or size < 0:
            raise ValueError("Invalid frame size")
        with self.guard():
            if size > self.limit - self.spent.value - self.reserved.value:
                self.denied.value += 1
                raise ByteBudgetExceeded("Application frame byte budget exhausted")
            self.reserved.value += size

    def sent(self, size):
        with self.guard():
            if size < 0 or size > self.reserved.value:
                raise ValueError("Invalid reserved-byte settlement")
            self.reserved.value -= size
            self.spent.value += size

    def release(self, size):
        with self.guard():
            if size < 0 or size > self.reserved.value:
                raise ValueError("Invalid unused-byte release")
            self.reserved.value -= size

    def state(self):
        with self.guard():
            return dict(
                limit=self.limit,
                spent=self.spent.value,
                reserved=self.reserved.value,
                denied_frames=self.denied.value,
            )


@contextmanager
def reserve_frame(budget, size):
    """Yield a settlement callback scoped to one frame reservation."""
    if budget is None:
        yield lambda count: None
        return
    budget.reserve(size)
    unsent = size

    def sent(count):
        nonlocal unsent
        if type(count) is not int or not 0 <= count <= unsent:
            raise ValueError("Sent bytes exceed this frame reservation")
        budget.sent(count)
        unsent -= count

    try:
        yield sent
    finally:
        budget.release(unsent)

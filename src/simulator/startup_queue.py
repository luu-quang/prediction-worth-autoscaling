from dataclasses import dataclass


@dataclass(frozen=True)
class BootBatch:
    ready_step: int
    count: int


class StartupQueue:
    """
    Tracks servers that are booting and become usable
    after a fixed provisioning delay.
    """

    def __init__(self, delay_steps: int):
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")

        self.delay_steps = delay_steps
        self._queue: list[BootBatch] = []

    @property
    def pending_count(self) -> int:
        return sum(batch.count for batch in self._queue)

    def schedule(self, current_step: int, count: int) -> None:
        """
        Schedule `count` servers to finish booting after delay_steps.

        Example:
        current_step = 4
        delay_steps = 2
        -> servers become ready at step 6
        """

        if current_step < 0:
            raise ValueError("current_step must be non-negative")

        if count < 0:
            raise ValueError("count must be non-negative")

        if count == 0:
            return

        ready_step = current_step + self.delay_steps

        self._queue.append(
            BootBatch(
                ready_step=ready_step,
                count=count,
            )
        )

    def complete_ready(self, current_step: int) -> int:
        """
        Return the number of servers whose boot delay
        has completed by current_step.

        Completed batches are removed from the queue.
        """

        if current_step < 0:
            raise ValueError("current_step must be non-negative")

        completed = 0
        remaining = []

        for batch in self._queue:
            if batch.ready_step <= current_step:
                completed += batch.count
            else:
                remaining.append(batch)

        self._queue = remaining

        return completed
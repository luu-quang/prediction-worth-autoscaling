import pytest

from src.simulator.startup_queue import StartupQueue


def test_servers_complete_after_delay():
    queue = StartupQueue(delay_steps=2)

    queue.schedule(current_step=0, count=3)

    assert queue.complete_ready(0) == 0
    assert queue.complete_ready(1) == 0
    assert queue.complete_ready(2) == 3


def test_pending_count():
    queue = StartupQueue(delay_steps=3)

    queue.schedule(current_step=0, count=2)
    queue.schedule(current_step=1, count=4)

    assert queue.pending_count == 6


def test_multiple_boot_batches():
    queue = StartupQueue(delay_steps=2)

    queue.schedule(current_step=0, count=2)
    queue.schedule(current_step=1, count=3)

    assert queue.complete_ready(1) == 0

    assert queue.complete_ready(2) == 2
    assert queue.pending_count == 3

    assert queue.complete_ready(3) == 3
    assert queue.pending_count == 0


def test_zero_delay_is_immediate():
    queue = StartupQueue(delay_steps=0)

    queue.schedule(current_step=5, count=4)

    assert queue.complete_ready(5) == 4


def test_invalid_delay_rejected():
    with pytest.raises(ValueError):
        StartupQueue(delay_steps=-1)
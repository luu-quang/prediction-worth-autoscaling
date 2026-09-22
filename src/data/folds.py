from dataclasses import dataclass

import pandas as pd

# Locked walk-forward design (docs/OPEN_DECISIONS_CLOSED.md, section E1):
# for test day k, train = days 1..k-2, validation = day k-1.
# Day 3 is excluded from primary analysis (too little training data).
PRIMARY_TEST_DAYS = (4, 5, 6, 7)

SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class Fold:
    fold_id: str
    test_day: int
    val_day: int
    train_days: tuple


def generate_folds(test_days=PRIMARY_TEST_DAYS):
    """Pre-registered walk-forward folds. Never a random train/test split."""
    folds = []
    for k in test_days:
        if k < 3:
            raise ValueError(f"test day {k} has no valid train/val days before it")
        folds.append(
            Fold(
                fold_id=f"fold_{k}",
                test_day=k,
                val_day=k - 1,
                train_days=tuple(range(1, k - 1)),
            )
        )
    return folds


def assign_day_index(timestamps: pd.Series) -> pd.Series:
    """Day 1 starts at the first timestamp in the series; days are 86400s blocks."""
    t0 = timestamps.min()
    elapsed_seconds = (timestamps - t0).dt.total_seconds()
    return (elapsed_seconds // SECONDS_PER_DAY).astype(int) + 1

import pandas as pd

from src.data.folds import assign_day_index, generate_folds


def test_generate_folds_matches_locked_spec():
    folds = generate_folds()
    assert [f.fold_id for f in folds] == ["fold_4", "fold_5", "fold_6", "fold_7"]

    by_id = {f.fold_id: f for f in folds}

    assert by_id["fold_4"].train_days == (1, 2)
    assert by_id["fold_4"].val_day == 3
    assert by_id["fold_4"].test_day == 4

    assert by_id["fold_7"].train_days == (1, 2, 3, 4, 5)
    assert by_id["fold_7"].val_day == 6
    assert by_id["fold_7"].test_day == 7


def test_generate_folds_never_reuses_day_3_as_test():
    folds = generate_folds()
    assert all(f.test_day != 3 for f in folds)


def test_assign_day_index_on_synthetic_trace(synthetic_demand_df):
    day_index = assign_day_index(synthetic_demand_df["timestamp"])
    assert day_index.min() == 1
    assert day_index.max() == 7
    # 288 five-minute steps per day at dt=300s
    assert (day_index.value_counts() == 288).all()


def test_assign_day_index_first_timestamp_is_day_1():
    timestamps = pd.Series(pd.date_range("2024-06-01", periods=5, freq="300s"))
    day_index = assign_day_index(timestamps)
    assert (day_index == 1).all()

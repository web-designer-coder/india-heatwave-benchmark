import pandas as pd
import pytest

from heatwave_benchmark.config import ExperimentConfig
from heatwave_benchmark.splits import (
    add_spatial_block_ids,
    buffered_spatiotemporal_split,
    make_spatial_block_folds,
    temporal_split,
)


def test_add_spatial_block_ids_is_deterministic_and_uses_fixed_degree_tiles():
    frame = pd.DataFrame(
        {"latitude": [7.5, 11.0, 12.5, 7.5], "longitude": [67.5, 69.0, 67.5, 72.5]}
    )

    result = add_spatial_block_ids(frame, block_degrees=5.0)

    assert result["spatial_block"].tolist() == [
        "lat1_lon13",
        "lat2_lon13",
        "lat2_lon13",
        "lat1_lon14",
    ]
    assert "spatial_block" not in frame.columns


def test_temporal_split_produces_disjoint_chronological_periods():
    frame = pd.DataFrame({"date": pd.date_range("1981-01-01", "2025-12-31", freq="YS")})

    split = temporal_split(frame, ExperimentConfig())

    assert split["train"]["date"].dt.year.min() == 1981
    assert split["train"]["date"].dt.year.max() == 2007
    assert split["validation"]["date"].dt.year.min() == 2008
    assert split["validation"]["date"].dt.year.max() == 2013
    assert split["test"]["date"].dt.year.min() == 2014
    assert split["test"]["date"].dt.year.max() == 2025


def test_spatial_block_folds_are_deterministic_and_cover_each_block_once():
    frame = pd.DataFrame({"spatial_block": [f"b{i}" for i in range(10)]})

    first = make_spatial_block_folds(frame, n_splits=5, seed=17)
    second = make_spatial_block_folds(frame, n_splits=5, seed=17)

    assert first == second
    flattened = [block for fold in first for block in fold]
    assert sorted(flattened) == [f"b{i}" for i in range(10)]
    assert len(flattened) == len(set(flattened))


def test_spatial_block_folds_reject_too_many_folds():
    with pytest.raises(ValueError, match="n_splits"):
        make_spatial_block_folds(pd.DataFrame({"spatial_block": ["a", "b"]}), n_splits=3)


def test_buffered_spatiotemporal_split_excludes_heldout_block_and_neighbor_buffer():
    years = range(1981, 2026)
    points = [
        ("test", 10.0, 70.0),
        ("buffer", 10.0, 71.0),
        ("train", 10.0, 73.1),
    ]
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(year=year, month=5, day=1),
                "grid_id": grid_id,
                "latitude": latitude,
                "longitude": longitude,
                "spatial_block": block,
                "target": year,
            }
            for year in years
            for grid_id, latitude, longitude in points
            for block in ["held" if grid_id == "test" else "neighbor" if grid_id == "buffer" else "far"]
        ]
    )

    split = buffered_spatiotemporal_split(
        frame,
        config=ExperimentConfig(),
        held_out_blocks=("held",),
        buffer_degrees=1.0,
    )

    assert set(split["train"]["grid_id"]) == {"train"}
    assert set(split["validation"]["grid_id"]) == {"train"}
    assert set(split["test"]["grid_id"]) == {"test"}
    assert split["train"]["date"].dt.year.max() < split["validation"]["date"].dt.year.min()
    assert split["validation"]["date"].dt.year.max() < split["test"]["date"].dt.year.min()


def test_buffered_spatiotemporal_split_rejects_unknown_heldout_blocks():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-05-01"]),
            "latitude": [10.0],
            "longitude": [70.0],
            "spatial_block": ["known"],
        }
    )

    with pytest.raises(ValueError, match="held_out_blocks"):
        buffered_spatiotemporal_split(
            frame,
            config=ExperimentConfig(),
            held_out_blocks=("unknown",),
        )

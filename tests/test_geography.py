import numpy as np

from heatwave_benchmark.geography import india_grid_mask, is_point_in_india


def test_india_boundary_includes_pune_and_excludes_adjacent_countries():
    assert is_point_in_india(18.5, 73.5)
    assert not is_point_in_india(19.5, 95.5)  # Myanmar
    assert not is_point_in_india(24.5, 90.5)  # Bangladesh


def test_india_grid_mask_uses_latitude_rows_and_longitude_columns():
    mask = india_grid_mask([18.5, 19.5], [73.5, 95.5])

    np.testing.assert_array_equal(mask.values, np.array([[True, False], [True, False]]))
    assert mask.dims == ("latitude", "longitude")

\n
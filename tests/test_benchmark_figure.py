import numpy as np
import pytest

from scripts.run_benchmark import _calibration_axis_limit


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ([0.03, 0.16, 0.184], 0.2),
        ([0.7, 0.96], 1.0),
        ([0.0, np.nan], 0.1),
    ],
)
def test_calibration_axis_limit_covers_binned_data_with_readable_minimum(probabilities, expected):
    assert _calibration_axis_limit(probabilities) == expected

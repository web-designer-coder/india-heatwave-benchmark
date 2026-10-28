import pytest

from heatwave_benchmark.config import ExperimentConfig


def test_default_config_uses_valid_forecast_horizons_and_chronological_periods():
    config = ExperimentConfig()

    assert config.forecast_horizons == (1, 2, 3)
    assert config.train_years == (1981, 2007)
    assert config.validation_years == (2008, 2013)
    assert config.test_years == (2014, 2025)
    assert config.label_normal_years == (1981, 2010)
    assert config.train_years[1] < config.validation_years[0]
    assert config.validation_years[1] < config.test_years[0]
    assert config.label_normal_years[1] < config.test_years[0]


def test_config_rejects_horizon_outside_supported_short_leads():
    with pytest.raises(ValueError, match="forecast_horizons"):
        ExperimentConfig(forecast_horizons=(1, 4))


def test_config_rejects_overlapping_time_partitions():
    with pytest.raises(ValueError, match="chronological"):
        ExperimentConfig(
            train_years=(1981, 2007),
            validation_years=(2006, 2010),
        )


def test_config_is_immutable():
    config = ExperimentConfig()

    with pytest.raises((AttributeError, TypeError)):
        config.max_lag_days = 2

"""Immutable configuration and validation for research experiments."""

from dataclasses import dataclass


YearRange = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Periods and forecast settings fixed before model evaluation."""

    label_normal_years: YearRange = (1981, 2010)
    train_years: YearRange = (1981, 2007)
    validation_years: YearRange = (2008, 2013)
    test_years: YearRange = (2014, 2025)
    forecast_horizons: tuple[int, ...] = (1, 2, 3)
    max_lag_days: int = 7
    random_seed: int = 42

    def __post_init__(self) -> None:
        for name in ("label_normal_years", "train_years", "validation_years", "test_years"):
            years = getattr(self, name)
            if (
                len(years) != 2
                or not all(isinstance(year, int) for year in years)
                or years[0] > years[1]
            ):
                raise ValueError(f"{name} must be an inclusive (start_year, end_year) pair")

        if not self.forecast_horizons or any(
            not isinstance(horizon, int) or horizon not in (1, 2, 3)
            for horizon in self.forecast_horizons
        ):
            raise ValueError("forecast_horizons must contain only supported leads 1, 2, or 3")
        if tuple(sorted(set(self.forecast_horizons))) != self.forecast_horizons:
            raise ValueError("forecast_horizons must be unique and in ascending order")
        if self.max_lag_days < 1:
            raise ValueError("max_lag_days must be positive")

        if not (
            self.train_years[1] < self.validation_years[0]
            and self.validation_years[1] < self.test_years[0]
        ):
            raise ValueError("train, validation, and test periods must be strictly chronological")
        if self.label_normal_years[1] >= self.test_years[0]:
            raise ValueError("label normal period must end before the final test period")

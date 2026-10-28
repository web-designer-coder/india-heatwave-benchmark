from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_SOURCES = frozenset({"nasa_power", "era5"})


def _validate_predictor_source(predictor_source: str) -> None:
    if predictor_source not in PREDICTOR_SOURCES:
        raise ValueError(f"unknown predictor_source: {predictor_source!r}")


def benchmark_output_directory(predictor_source: str) -> Path:
    _validate_predictor_source(predictor_source)
    return PROJECT_ROOT / "reports" / f"generated_{predictor_source}"


def benchmark_feature_frame_path(predictor_source: str) -> Path:
    _validate_predictor_source(predictor_source)
    return PROJECT_ROOT / "data" / "interim" / f"benchmark_features_{predictor_source}.csv.gz"


def transition_output_directory(predictor_source: str) -> Path:
    _validate_predictor_source(predictor_source)
    return PROJECT_ROOT / "reports" / f"generated_state_transition_{predictor_source}"

import pytest

from heatwave_benchmark.artifacts import (
    benchmark_feature_frame_path,
    benchmark_output_directory,
    transition_output_directory,
)


def test_generated_artifacts_are_scoped_by_predictor_source():
    nasa_output = benchmark_output_directory("nasa_power")
    era5_output = benchmark_output_directory("era5")
    nasa_features = benchmark_feature_frame_path("nasa_power")
    era5_features = benchmark_feature_frame_path("era5")

    assert nasa_output.name == "generated_nasa_power"
    assert era5_output.name == "generated_era5"
    assert nasa_features.name == "benchmark_features_nasa_power.csv.gz"
    assert era5_features.name == "benchmark_features_era5.csv.gz"
    assert nasa_output != era5_output
    assert nasa_features != era5_features
    assert transition_output_directory("nasa_power").name == "generated_state_transition_nasa_power"
    assert transition_output_directory("era5").name == "generated_state_transition_era5"


@pytest.mark.parametrize("source", ["", "merra2", "ERA5"])
def test_artifact_path_helpers_reject_unknown_predictor_sources(source):
    with pytest.raises(ValueError, match="predictor_source"):
        benchmark_output_directory(source)

\n
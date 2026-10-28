"""Download and validate public IMD Tmax and retrospective daily predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heatwave_benchmark.data import (  # noqa: E402
    ERA5_ARCHIVE_API,
    ERA5_DAILY_VARIABLES,
    ERA5_ICECHUNK_REGISTRY,
    IMD_TMAX_DOWNLOAD,
    NASA_POWER_DAILY_API,
    NASA_POWER_PARAMETERS,
    build_file_manifest,
    download_imd_tmax_year,
    download_era5_seasonal_grid,
    download_era5_icechunk_seasonal_grid,
    download_nasa_power_grid,
    load_imd_terrain_mask,
)
from heatwave_benchmark.pipeline import (  # noqa: E402
    assemble_era5_dataset,
    assemble_power_dataset,
    read_imd_archive,
    select_analysis_points,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--predictor-source",
        choices=("nasa_power", "era5"),
        default="nasa_power",
        help="Retrospective daily predictor archive (default: nasa_power)",
    )
    parser.add_argument(
        "--era5-provider",
        choices=("icechunk", "open-meteo"),
        default="icechunk",
        help="ERA5 data delivery source (default: public Icechunk S3 archive)",
    )
    parser.add_argument("--max-workers", type=int, choices=range(1, 6), default=1, help="Concurrent NASA POWER point requests (maximum five)")
    parser.add_argument("--batch-size", type=int, default=5, help="ERA5 locations per API request (1-5)")
    parser.add_argument("--request-interval", type=float, default=1.0, help="Seconds between ERA5 batch requests")
    parser.add_argument(
        "--era5-workers",
        type=int,
        choices=range(1, 9),
        default=4,
        help="Concurrent independent ERA5 Icechunk spatial tiles (1-8)",
    )
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if args.start_year < 1981 or args.end_year > 2025 or args.start_year > args.end_year:
        parser.error("the configured study period must be an ordered subset of 1981-2025")

    raw_imd = PROJECT_ROOT / "data" / "raw" / "imd"
    predictor_cache_name = (
        ("era5_icechunk" if args.era5_provider == "icechunk" else "era5_seasonal")
        if args.predictor_source == "era5"
        else args.predictor_source
    )
    raw_predictors = PROJECT_ROOT / "data" / "raw" / predictor_cache_name
    interim = PROJECT_ROOT / "data" / "interim"
    processed = PROJECT_ROOT / "data" / "processed"
    for directory in (raw_imd, raw_predictors, interim, processed):
        directory.mkdir(parents=True, exist_ok=True)

    imd_records = []
    with requests.Session() as session:
        for year in range(args.start_year, args.end_year + 1):
            last_error = None
            for attempt in range(args.retries + 1):
                try:
                    path = download_imd_tmax_year(year, raw_imd, session=session)
                    imd_records.append(
                        build_file_manifest(
                            path,
                            source_url=IMD_TMAX_DOWNLOAD,
                            dataset="IMD daily 1-degree Tmax grid",
                            year=year,
                            resolution_degrees=1.0,
                        )
                    )
                    print(f"IMD {year}: validated ({path.stat().st_size:,} bytes)", flush=True)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == args.retries:
                        raise RuntimeError(f"IMD download failed for {year}") from exc
                    time.sleep(1.5 * (2**attempt))
    (raw_imd / "download_manifest.json").write_text(
        json.dumps(imd_records, indent=2), encoding="utf-8"
    )

    tmax = read_imd_archive(raw_imd, args.start_year, args.end_year).astype("float32")
    terrain_mask = load_imd_terrain_mask()
    domain_start_year = max(args.start_year, 1981)
    domain_end_year = min(args.end_year, 2010)
    if domain_start_year > domain_end_year:
        domain_start_year, domain_end_year = args.start_year, args.end_year
    observation_period = (
        f"{domain_start_year}-01-01",
        f"{domain_end_year}-12-31",
    )
    points = select_analysis_points(
        tmax,
        terrain_mask,
        observation_period=observation_period,
    )
    points_path = interim / "analysis_grid_points.csv"
    points.to_csv(points_path, index=False)
    print(f"Selected {len(points)} observed land grid cells", flush=True)
    tmax_path = interim / f"imd_tmax_{args.start_year}_{args.end_year}.nc"
    tmax.to_dataset().to_netcdf(tmax_path, engine="scipy")
    print(f"Saved IMD analysis field: {tmax_path}", flush=True)

    if args.predictor_source == "nasa_power":
        predictor_manifest = download_nasa_power_grid(
            points,
            raw_predictors,
            start_date=f"{args.start_year}-01-01",
            end_date=f"{args.end_year}-12-31",
            max_workers=args.max_workers,
            retries=args.retries,
            retry_delay_seconds=args.request_interval,
        )
        predictor_dataset = assemble_power_dataset(
            raw_predictors,
            points,
            start_date=f"{args.start_year}-01-01",
            end_date=f"{args.end_year}-12-31",
        )
        predictor_filename = f"nasa_power_{args.start_year}_{args.end_year}.nc"
        predictor_api = NASA_POWER_DAILY_API
        predictor_model = "NASA POWER daily meteorology; cached API source tags retained"
        predictor_variables = {name.lower(): name for name in NASA_POWER_PARAMETERS}
        predictor_time_standard = "LST"
        predictor_cell_selection = "IMD 1-degree cell centers queried through POWER point API; response source coordinates retained"
        predictor_native_resolution = "MERRA-2 meteorology: 0.5 degree latitude by 0.625 degree longitude"
    else:
        if args.era5_provider == "icechunk":
            predictor_manifest = download_era5_icechunk_seasonal_grid(
                points,
                raw_predictors,
                start_year=args.start_year,
                end_year=args.end_year,
                max_workers=args.era5_workers,
                progress_callback=lambda year_index, year_total, year, current, total, complete, error: print(
                    f"ERA5 Icechunk window {year} ({year_index}/{year_total}): "
                    f"{complete}/{len(points)} points validated"
                    + (f"; error: {error}" if error else ""),
                    flush=True,
                ),
            )
            predictor_api = ERA5_ICECHUNK_REGISTRY
            predictor_model = "Copernicus ERA5 via Earthmover public Icechunk temporal layout"
            predictor_variables = {
                "t2m": "temperature_2m_mean/max/min",
                "d2m": "hourly dewpoint for Magnus relative humidity",
                "u10": "10 m eastward wind component",
                "v10": "10 m northward wind component",
                "sp": "surface pressure",
                "tp": "hourly total precipitation accumulation",
            }
            predictor_time_standard = "UTC hourly; aggregated to Asia/Kolkata local dates"
            predictor_cell_selection = "nearest 0.25-degree ERA5 grid cell to each IMD 1-degree cell center"
            predictor_native_resolution = "ERA5 reanalysis: 0.25 degree"
        else:
            predictor_manifest = download_era5_seasonal_grid(
                points,
                raw_predictors,
                start_year=args.start_year,
                end_year=args.end_year,
                max_workers=1,
                retries=args.retries,
                batch_size=args.batch_size,
                request_interval_seconds=args.request_interval,
                progress_callback=lambda year_index, year_total, year, current, total, complete, error: print(
                    f"ERA5 Open-Meteo window {year} ({year_index}/{year_total}), batch {current}/{total}: "
                    f"{complete}/{len(points)} points validated"
                    + (f"; error: {error}" if error else ""),
                    flush=True,
                ),
            )
            predictor_api = ERA5_ARCHIVE_API
            predictor_model = "ERA5 via Open-Meteo Historical Weather API"
            predictor_variables = {feature: variable for variable, feature in ERA5_DAILY_VARIABLES}
            predictor_time_standard = "Asia/Kolkata"
            predictor_cell_selection = "nearest ERA5 grid cell; elevation downscaling disabled"
            predictor_native_resolution = "ERA5 reanalysis: 0.25 degree"
        predictor_dataset = assemble_era5_dataset(
            raw_predictors,
            points,
            start_date=f"{args.start_year}-01-01",
            end_date=f"{args.end_year}-12-31",
        )
        predictor_filename = f"era5_{args.start_year}_{args.end_year}.nc"
    failures = predictor_manifest.loc[~predictor_manifest["status"].isin(("cached", "downloaded"))]
    if not failures.empty:
        print(
            f"Predictor retrieval is incomplete for {len(failures)} grid cells; "
            "resolve the reported source error and rerun to resume validated annual caches.",
            flush=True,
        )
        columns = [column for column in ("grid_id", "incomplete_year", "status", "error") if column in failures]
        print(failures[columns].to_string(index=False), flush=True)
        return 2
    print(
        f"{args.predictor_source} points: {len(predictor_manifest)} complete; "
        f"{(predictor_manifest.status == 'cached').sum()} reused from cache",
        flush=True,
    )

    predictor_path = interim / predictor_filename
    predictor_dataset.to_netcdf(predictor_path, engine="scipy")
    first_raw_file = raw_predictors / f"{predictor_manifest.iloc[0]['grid_id']}.json"
    raw_header = json.loads(first_raw_file.read_text(encoding="utf-8")).get("header", {}) if first_raw_file.exists() else {}
    source_manifest_file = raw_predictors / "source_manifest.json"
    source_manifest = (
        json.loads(source_manifest_file.read_text(encoding="utf-8"))
        if source_manifest_file.exists()
        else {}
    )

    acquisition = {
        "accessed_on": date.today().isoformat(),
        "period": [args.start_year, args.end_year],
        "imd_manifest": str((raw_imd / "download_manifest.json").relative_to(PROJECT_ROOT)),
        "imd_netcdf": str(tmax_path.relative_to(PROJECT_ROOT)),
        "predictor_source": args.predictor_source,
        "predictor_delivery": args.era5_provider if args.predictor_source == "era5" else "NASA POWER API",
        "predictor_api": predictor_api,
        "predictor_api_version": raw_header.get("api", {}).get("version", ""),
        "predictor_snapshot_id": source_manifest.get("snapshot_id", ""),
        "predictor_source_manifest": (
            str(source_manifest_file.relative_to(PROJECT_ROOT))
            if source_manifest_file.exists()
            else ""
        ),
        "predictor_data_license": source_manifest.get("license", ""),
        "predictor_model": predictor_model,
        "predictor_variables": predictor_variables,
        "predictor_time_standard": predictor_time_standard,
        "predictor_cell_selection": predictor_cell_selection,
        "predictor_native_resolution": predictor_native_resolution,
        "predictor_grid_points": int(len(points)),
        "analysis_domain_observation_period": list(observation_period),
        "predictor_manifest": str((raw_predictors / "download_manifest.csv").relative_to(PROJECT_ROOT)),
        "predictor_netcdf": str(predictor_path.relative_to(PROJECT_ROOT)),
        "terrain_mask": "imdlib 0.4.0 region_mask_100.npy, MIT license",
        "terrain_mask_sha256": hashlib.sha256(
            (PROJECT_ROOT / "src" / "heatwave_benchmark" / "imd_region_mask_1deg.csv").read_bytes()
        ).hexdigest(),
        "country_mask": "Natural Earth Admin 0 - Countries 1:10m version 5.1.1; grid-center inside India polygon",
        "country_boundary_source": "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
        "country_boundary_source_archive_sha256": "ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6",
        "country_boundary_geojson_sha256": hashlib.sha256(
            (PROJECT_ROOT / "src" / "heatwave_benchmark" / "india_boundary_10m.geojson").read_bytes()
        ).hexdigest(),
        "label_grid_resolution_degrees": 1.0,
        "notes": "The predictor archive is retrospective meteorological analysis/reanalysis data; results are retrospective prediction experiments and not operational forecast skill.",
    }
    (processed / "acquisition_manifest.json").write_text(
        json.dumps(acquisition, indent=2), encoding="utf-8"
    )
    print(f"Saved predictor field: {predictor_path}", flush=True)
    print(f"Saved acquisition manifest: {processed / 'acquisition_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

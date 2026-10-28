# Focused novelty search log (27 October 2026)

## Database-backed search

An initial OpenAlex Works API search was run on 25 October 2026. The script `scripts/search_openalex_heatwave.py` submitted nine exact full-text searches and requested the first 50 relevance-ranked records per query. It returned 423 records in total and 335 unique OpenAlex works after deduplication. The returned metadata and available abstracts are retained in `reports/openalex_heatwave_search_2026-10-25.json`; the query script and a console summary of India-relevant candidates are `scripts/search_openalex_heatwave.py` and `scripts/summarize_openalex_candidates.py`. OpenAlex's ranking and per-query cap make this a focused search, not an exhaustive bibliographic census.

The script `scripts/document_novelty_search.py` queried the Crossref REST API on 27 October 2026 with these exact bibliographic queries:

1. `India heatwave prediction machine learning grid`
2. `Indian heatwave onset machine learning`
3. `India heatwave forecast spatial validation grid cell`
4. `IMD gridded heatwave event probability prediction India`

The first 30 results for each query, including DOI and available title, author, publication, and journal metadata, are saved in `docs/novelty_search_crossref_2026-10-27.json`. This record preserves the search inputs and returned metadata; it is not a systematic review and the result cap means it does not establish an exhaustive set of papers.

## Targeted publisher search and screening

ScienceDirect searches used the terms `Indian heatwave prediction machine learning`, `India heatwave onset prediction spatial validation`, and `India heatwave forecast machine learning gridded`. Relevant full-text or publisher records screened included:

- Bhoopathi et al. (2024), short-range Tmax and heatwave-day forecasting across four Indian temperature zones, DOI [10.1016/j.acags.2024.100204](https://doi.org/10.1016/j.acags.2024.100204).
- Bhoopathi et al. (2023), maximum-temperature prediction associated with heatwave thresholds in Rajasthan and Karnataka, DOI [10.1016/j.rsase.2023.101048](https://doi.org/10.1016/j.rsase.2023.101048).
- Budamala et al. (2026), adaptive post-processing of India-wide operational subseasonal-to-seasonal heatwave forecasts, DOI [10.1016/j.hydroa.2026.100221](https://doi.org/10.1016/j.hydroa.2026.100221).
- Singh et al. (2026), machine-learning post-processing of operational NCMRWF station temperature forecasts, DOI [10.1039/D6VA00077K](https://doi.org/10.1039/D6VA00077K).
- Choudary et al. (2025), heatwave-event classification in a Chennai station series, DOI [10.1038/s41598-025-04634-9](https://doi.org/10.1038/s41598-025-04634-9).
- Rao et al. (2023), 10-day machine-learning prediction of daily Tmax anomalies over India, DOI [10.1038/s41598-023-44286-1](https://doi.org/10.1038/s41598-023-44286-1).

These records establish that Indian heatwave and temperature prediction with machine learning, India-wide gridded/event evaluation, and operational forecast post-processing already have substantial precedents. The manuscript therefore claims neither the first Indian heatwave-prediction benchmark nor a new forecasting algorithm. It distinguishes its scope by the IMD-aligned one-degree binary cell target, buffered geographic holdouts, state-conditioned onset diagnostic, and a frozen temporal evaluation. These are descriptive differences, not proof of priority or novelty.

## Limits and remaining review

Crossref metadata ranking and publisher search can miss relevant papers, especially non-English work, conference proceedings, theses, and papers whose metadata omit these terms. Scopus and Web of Science were not queried, and no full-text systematic screening protocol was applied. Before asserting a stronger novelty claim, a subject librarian or domain researcher should extend this search across Scopus/Web of Science and review primary studies. A meteorology/climate-science expert must separately review the heatwave rule, station-coverage change interpretation, and station-check caveats; this search is not that expert review.

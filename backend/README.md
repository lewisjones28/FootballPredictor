# Backend

This document describes the backend package for the FootballPredictor project.
It explains package layout, how to install dependencies, how to run the main
scripts and where input/output files live.

---

## Quick summary

- The backend package lives in `backend/` and contains modular subpackages:
  - `backend/io` - I/O utilities (downloader)
  - `backend/data` - data loaders and helpers
  - `backend/analysis` - analysis utilities (team statistics)
  - `backend/models` - trainer, prediction engine, result generator
  - `backend/utils` - constants and configuration
- Entrypoints:
  - CLI wrapper: `backend/football_predictions.py` (calls the modular `main()`)

## Project layout (key files)

- `backend/football_predictions.py` — thin CLI wrapper (use `python -m backend.football_predictions`)
- `backend/io/downloader.py` — downloads fixture CSVs
- `backend/analysis/team_analyzer.py` — `TeamAnalyzer` for team form/strength metrics
- `backend/models/trainer.py` — `ModelTrainer` that creates training frames
- `backend/models/engine.py` — `PredictionEngine`, `ResultGenerator`, worker functions and `main()`
- `backend/utils/constants.py` — column names and constants
- `backend/utils/config.py` — `PredictionConfig` dataclass and `ConfigurationManager`
- `backend/files/` — input and output CSVs (input tracked; output ignored by default in `.gitignore`)

## Prerequisites

- Python 3.10+ recommended
- A virtual environment (venv/conda) is strongly recommended

Install dependencies from the repo root:

```bash
python -m venv .venv            # create virtual env (optional)
source .venv/bin/activate       # macOS / Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` contains the core dependencies used by the backend (pandas, numpy, scikit-learn, requests, beautifulsoup4, etc.).

## Input and output data

- Inputs (fixture CSVs): `backend/files/input/<league>/<year>.csv` (example: `backend/files/input/epl/2025.csv`).
- Outputs (aggregated predictions): `backend/files/output/<league>/<year>/<round>/final_predictions.csv`.

## Where the data comes from

- Input data is CSV (spreadsheet-style) files placed under `backend/files/input/<league>/<year>.csv` (for example: `backend/files/input/epl/2025.csv`).
- The project includes a small downloader utility that can fetch CSVs from a public source when available.

### Downloader — how it works

The downloader is implemented in `backend/io/downloader.py` as the function `download_fixtures(league_name, year)`.

High-level behavior:

- It constructs a page URL using the requested league and year, in the form:

```text
https://fixturedownload.com/download/csv/{league}-{year}
```

- The downloader fetches that page and parses the HTML to locate the direct CSV download link. The code looks for an anchor tag whose text is exactly:

```
click here to download
```

- Once that link is found, the downloader builds the direct CSV URL by prefixing the link path with the site host, e.g.: 

```python
csv_url = "https://fixturedownload.com" + link['href']
```

- The CSV is then downloaded (using `requests`) and saved to:

```
backend/files/input/<league>/<year>.csv
```

- Before saving, the downloader attempts a lightweight optimization: it reads the `Content-Length` header from the CSV response and compares that size to any existing local file. If the local file size is greater than or equal to the remote `Content-Length`, the download is skipped to avoid redundant downloads.

- The downloader creates the necessary directories if they do not exist and logs progress via the standard logging facility.

This ensures the CSV is present under `backend/files/input/...` before the rest of the pipeline runs.

Notes and caveats

- The downloader relies on the third-party site `fixturedownload.com` and the HTML structure referenced above; if that site changes, the parser may no longer find the link.
- The downloader requires network access and the `requests` and `beautifulsoup4` packages to be installed.

## Running the backend

Preferred (module style) — run the CLI wrapper from the repository root:

```bash
python -m backend.football_predictions
```

This will prompt for configuration (league, year, iterations, rounds to predict) and then execute the prediction pipeline.
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

---

## Flask API (new)

A lightweight Flask API is included to serve aggregated predictions from `backend/files/output` and make them available to frontend clients (or other services) without requiring direct filesystem access.

### Purpose

- Load `final_predictions.csv` files from `backend/files/output/<league>/<year>/<round>/final_predictions.csv` into memory on startup.
- Provide REST endpoints to list available leagues/years/rounds and to fetch predictions for a specific league/year/round.
- Expose a `/api/reload` endpoint to reload prediction files from disk without restarting the service.
- Enable CORS so the frontend can fetch data from the API.

### Where the code lives

- `backend/api.py` — Flask application that implements the endpoints and the in-memory cache.
- The loader discovers prediction files automatically, loads CSVs into an in-memory dictionary and serves JSON responses.

### Endpoints

- `GET /api/health` — simple health check. Returns `{"status":"healthy","service":"Football Predictions API"}`.
- `GET /api/leagues` — returns available leagues (example: `{"leagues": ["epl"]}`).
- `GET /api/leagues/<league>/years` — returns available years for a league (example: `{"league":"epl","years":["2025"]}`).
- `GET /api/leagues/<league>/years/<year>/rounds` — returns available rounds for the league & year (example: `{"league":"epl","year":"2025","rounds":[20,21]}`).
- `GET /api/predictions/<league>/<year>/<round>` — returns predictions for a single round. Supports optional query `?team=TEAM_NAME` to filter by team (matches home or away). Response example:

```json
{
  "league": "epl",
  "year": "2025",
  "round": 20,
  "count": 6,
  "predictions": [
    {"Match Number":194, "Round Number":20, "Date":"04/01/2026 15:00", "Location":"Stadium", "Home Team":"Everton", "Away Team":"Brentford", "Result":"2 - 1", "Predicted": true},
    {"Match Number":194, "Round Number":20, "Date":"04/01/2026 15:00", "Location":"Stadium", "Home Team":"Everton", "Away Team":"Brentford", "Result":"2 - 1", "Predicted": false}
  ]
}
```

- `GET /api/predictions` — returns a summary of all available predictions grouped by league and year, including available rounds and total predictions per league/year.
- `POST /api/reload` — reload all predictions from disk into memory. Use this after regenerating prediction CSVs to refresh the API cache without restarting.

### Behavior & implementation notes

- The API discovers prediction files by walking `backend/files/output` and looking for `final_predictions.csv` under `/<league>/<year>/<round>/`.
- Files are parsed with `pandas.read_csv()` and the resulting records are stored in an in-memory cache (a nested dict keyed by league -> year -> round).
- The API includes both predicted rows and actual (non-predicted) results if they are present in the `final_predictions.csv`. When predictions are regenerated, calling `POST /api/reload` will refresh the cache.
- The initial load happens on startup (`api.py` calls the loader), so make sure output files exist before starting the API or call `/api/reload` once files are generated.

### Running the API

From the `backend/` directory you can run the API directly:

```bash
python api.py
```

- Default port: `5000`. If port 5000 is in use you can override with the `PORT` environment variable:

```bash
PORT=5001 python api.py
```

- The `start-all.sh` script in the repo root also launches the API (on port 5001 by default) and the frontend dev server together.

### Requirements

The API requires `flask` and `flask-cors` which have been added to `backend/requirements.txt`. Install/update the backend dependencies and activate your virtual environment prior to running the API.

### Example usage from the frontend

- The frontend is configured to call `http://localhost:5001/api` by default (see `frontend/.env`).
- Example request to fetch round 20 predictions for EPL:

```bash
curl http://localhost:5001/api/predictions/epl/2025/20
```

### Troubleshooting

- If the API returns empty lists, confirm prediction files exist at `backend/files/output/<league>/<year>/<round>/final_predictions.csv`.
- Use `POST /api/reload` after running the prediction engine to refresh the in-memory cache.
- If `PORT 5000` is already in use, the server will fail to bind; use `PORT=5001` (or another free port) to start the API.

---

## Notes

- The backend prediction CSVs (`backend/files/output/`) are intentionally not tracked in git. If you want to retain outputs in version control, remove that path from `.gitignore`.

## Running the backend (recap)

```bash
# Build the virtualenv, install deps and run the CLI wrapper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.football_predictions
```

Or run the API directly for other clients (frontend):

```bash
# From backend/ folder
python api.py
# or on alternate port
PORT=5001 python api.py
```

---

If you want, I can also add a short example script that shows how to call the API and persist selected predictions into a local file or demonstrate the `POST /api/reload` flow. Let me know which you'd prefer.

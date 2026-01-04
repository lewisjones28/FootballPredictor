# Football Predictor — Frontend

A React + TypeScript frontend (Vite) for viewing football match predictions produced by the backend.

This README explains how to run the app in development, configure environment variables, build for production, and troubleshoot common issues.

Table of contents
- Project overview
- Tech stack
- Features
- Project structure
- Environment variables
- Quick start (development)
- Build & preview (production)
- Linting & type checking
- How the frontend consumes the backend API
- Troubleshooting
- Deployment notes
- Contributing
- License

---

Project overview
----------------
The frontend is a small single-page application built with React + TypeScript and Vite. It reads aggregated prediction data (CSV converted to JSON) from the backend Flask API and displays match predictions by league and round in a dark, easy-to-read UI.

Tech stack
----------
- React 18 (TypeScript)
- Vite — dev server and build tool
- PapaParse (only used previously; replaced by backend API in this repo)
- CSS with CSS variables for theming

Features
--------
- Dark "cool" theme designed for comfortable viewing
- League & Round filters
- Responsive match prediction cards showing teams, predicted score, match date/time and venue
- Loading and error states
- CORS-friendly backend API consumption

Project structure
-----------------
Important files and folders:
```
frontend/
├─ src/
│  ├─ components/            # React components (PredictionCard, etc.)
│  ├─ types/                 # TypeScript interfaces
│  ├─ utils/                 # Data loader and helpers
│  ├─ App.tsx                # Main application
│  ├─ main.tsx               # React entry
│  ├─ index.css              # CSS variables (theme)
│  └─ App.css                # Component styles
├─ public/                   # Static assets
├─ index.html                # App HTML template
├─ package.json              # Scripts + dependencies
├─ tsconfig.json             # TypeScript config
├─ vite.config.ts            # Vite configuration
└─ .env                      # Environment variables for dev (not committed if added to .gitignore)
```

Environment variables
---------------------
The frontend reads the following environment variable from Vite's environment configuration. Create a `.env` file in the `frontend/` folder for local development.

- `VITE_API_URL` — Base URL for the backend API (including the `/api` path is optional). Default used by the app: `http://localhost:5001/api`.

Example `.env` file (frontend/.env):
```env
# Backend API base URL used by the frontend during development
VITE_API_URL=http://localhost:5001/api
```

Quick start (development)
-------------------------
Prerequisites:
- Node.js 18+ (or a compatible LTS release)
- npm (or a different package manager; commands below use npm)

Install dependencies and start the dev server:

```bash
cd frontend
npm install         # install dependencies (first time only)
npm run dev         # start Vite dev server (HMR)
```

Open your browser to the address printed by Vite (usually http://localhost:5173).

Notes:
- The frontend expects the backend API to be running (default: `http://localhost:5001/api`). If the backend is not available, the UI will show an error message and fallback lists may be used for leagues/rounds.
- The `.env` file is read by Vite at startup. If you change `VITE_API_URL`, restart the dev server to pick up the change.

Build & preview (production)
----------------------------
Build the optimized production bundle and preview it locally:

```bash
cd frontend
npm run build        # produces a dist/ folder
npm run preview      # serve the production build locally
```

The preview server typically serves the built site on http://localhost:4173 (port may vary).

Linting & type checking
-----------------------
Run TypeScript checks:
```bash
cd frontend
npx tsc --noEmit
```

Run linting (if ESLint configured):
```bash
npm run lint
```

How the frontend consumes the backend API
---------------------------------------
This frontend is designed to work with the Flask API implemented in `backend/api.py`.

- Base API URL: `VITE_API_URL` (defaults to `http://localhost:5001/api` in this repo).
- Important endpoints the frontend uses:
  - `GET /api/leagues` — list available leagues
  - `GET /api/leagues/:league/years/:year/rounds` — list rounds for a league/year
  - `GET /api/predictions/:league/:year/:round` — get predictions for a single round

Example API call (curl):
```bash
curl "${VITE_API_URL:-http://localhost:5001/api}/predictions/epl/2025/20"
```

If you run the backend API on a different port, update `VITE_API_URL` in `frontend/.env` and restart the dev server.

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

Deployment notes
----------------
The frontend builds a static bundle (`dist/`) which can be served by any static hosting (Netlify, Vercel, S3 + CloudFront, nginx, etc.).

If the backend API is hosted separately, make sure to set `VITE_API_URL` to the production API endpoint when building the frontend.

Example build & deploy flow:
```bash
# build with production API
VITE_API_URL=https://api.yourdomain.com/api npm run build
# then upload contents of dist/ to your static host
```

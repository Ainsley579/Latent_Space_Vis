# Latent Space Vis – Frontend–Backend Integration

This document describes how to run the backend and frontend and how they connect, plus an index of related READMEs.

---

## 1. Backend (Flask)

- **Entry:** `latentvis/backend/flask_backend.py`
- **Default URL:** `http://127.0.0.1:5001`

**Run:**

```bash
cd latentvis/backend
python flask_backend.py
```

**API reference:** Request/response formats, parameters, and error codes for all endpoints are in  
→ **[backend/README_API.md](backend/README_API.md)**

Summary of main endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | Check backend liveness |
| POST | `/search-smiles` | Look up real_ids by SMILES |
| POST | `/generate-range` | Filter Sankey data by epoch range |
| POST | `/recluster-agglomerative` | Re-run Agglomerative with chosen k and linkage and return new labels |

---

## 2. Frontend (Next.js)

- **Root:** `latentvis/frontend`
- **Backend base URL:** Set via `NEXT_PUBLIC_FLASK_API_BASE`; default is `http://127.0.0.1:5001` if unset.

**Run:**

```bash
cd latentvis/frontend
npm install
npm run dev
```

To use a different backend URL (e.g. another host or port):

```bash
NEXT_PUBLIC_FLASK_API_BASE=http://127.0.0.1:5001 npm run dev
```

The frontend calls the backend at `${NEXT_PUBLIC_FLASK_API_BASE}/<path>`, e.g.  
`${NEXT_PUBLIC_FLASK_API_BASE}/recluster-agglomerative`.

---

## 3. Running both (local development)

1. Start the backend first:
   ```bash
   cd latentvis/backend && python flask_backend.py
   ```
2. Start the frontend:
   ```bash
   cd latentvis/frontend && npm run dev
   ```
3. Open the URL shown by the frontend (e.g. `http://localhost:3000`). The frontend will send API requests to `http://127.0.0.1:5001` unless `NEXT_PUBLIC_FLASK_API_BASE` is set otherwise.

---

## 4. Related READMEs

| File | Description |
|------|-------------|
| **backend/README_API.md** | Flask API (request/response, errors, env vars) |
| **backend/README.md** | Backend pipeline, layout, script usage |
| **backend/README_PIPELINE.md** | Pipeline details (if present) |
| **README_INTEGRATION.md** (this file) | How to run and connect frontend and backend |

Add new READMEs to this table with path and a short description as needed.

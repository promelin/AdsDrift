# AGENTS.md — AdsDrift

## Project overview

AdsDrift is a Python ML research codebase for generating adsorption structures.
It is a **CLI/library project, not a web app** — there is no original web server.
A lightweight Flask dashboard (`web/`) was added to demonstrate the components
that work without external data.

## Running the app

```bash
docker compose -f docker-compose.base44.yml up -d --build
```

The dashboard is served on **port 3000**. Health check: `GET /health`.

## Architecture

- **Base image**: `python:3.12-slim` (Dockerfile.base44)
- **Source mount**: repo bind-mounted at `/workspace/AdsDrift`
- **PYTHONPATH**: `/workspace` (so `from AdsDrift.model...` resolves)
- **Web server**: Flask dev server with `debug=True` (auto-reload on `web/` file changes)
- **Live reload**: Flask reloader watches `web/` files; project code changes need `reload_preview`

## Dependencies

Installed in the Docker image (Dockerfile.base44):
- torch (CPU-only), e3nn, ase, numpy, scipy, matplotlib, PyYAML, flask

**Not installed** (only needed for training, not the demo):
- `mace-torch` — frozen MACE-MH-1 feature encoder used during training
- `matscipy` — neighbour list used by the MACE feature extractor

To enable training, add these to the Dockerfile and provide:
- MACE-MH-1 weights (set `ADSDRIFT_MACE_CHECKPOINT` env var)
- OC20-Dense training data (not distributed in this repo)
- Trained `test_18` checkpoints (not distributed; needed for `generate_pdb.py` inference)

## What works without external data

1. **Adsorbate library** (`/adsorbates`) — 86 bundled templates, no deps beyond ASE
2. **R0 random initialization** (`/sampling`) — stratified surface coverage sampling with rejection; creates slab from ASE bulk + adsorbate from library
3. **Generator architecture** (`/model`) — instantiate from any `config/*.yaml`, inspect parameter count

## Key import note

The repo root has an `__init__.py` and must be importable as the `AdsDrift` package.
In Docker, the repo is at `/workspace/AdsDrift` with `PYTHONPATH=/workspace`.
Locally, `web/app.py` creates a symlink `AdsDrift -> <repo_root>` in the parent dir.

## Verifying the app

```bash
curl http://localhost:3000/health          # → {"status":"ok"}
curl http://localhost:3000/adsorbates      # → HTML table with 86 entries
curl -X POST http://localhost:3000/api/sample \
  -H 'Content-Type: application/json' \
  -d '{"surface":"Cu(111)","adsorbate":"5","num_samples":50,"seed":42}'
```

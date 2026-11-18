# PathSnap – Map-Matching & Route Reconstruction (UE24CS352A ML Hackathon)

Noisy 1 Hz GPS trajectories are matched to a road network:
**candidate arcs → ML classifier scores each candidate → HMM/Viterbi picks a connected route →
FastAPI (Laptop 1) → browser UI (Laptop 2).**

## Repository layout
```
src/data_io.py         raw file loading, cleaning, per-trip local metric projection
src/pipeline.py        candidate generation (60 m, k=8), feature engineering, reference-label alignment
src/build_dataset.py   builds feature tables for train / student_test + trip inventory
src/eda.py             EDA tables, plots, Folium maps
src/metrics.py         candidate metrics, Top-1 accuracy, candidate miss rate, route continuity
src/train_models.py    5 models, GroupKFold CV on train trips, student-test evaluation, plots
src/hmm.py             HMM + Viterbi decoder
src/evaluate_hmm.py    model selection + classifier-only vs HMM comparison
api/main.py            FastAPI backend (Laptop 1)
ui/index.html          evaluator UI (Laptop 2), Leaflet map
results/               CSV results and logs      reports/figures, reports/maps   plots and Folium maps
models/                trained models (.joblib) + selected_model.json
```

## Data
Place the dataset folder **next to** this repo (default) or set `PATHSNAP_DATA`:
```
Downloads/
├── ml_hackathon/          (this repo)
└── Student folder/
    ├── train/<trip>/<trip>.{track,nodes,arcs,route}
    └── student_test/<trip>/...
```
File formats: `.track` = lon lat t ; `.nodes` = lon lat (row = node id) ; `.arcs` = from to (row = arc id) ;
`.route` = ordered reference arc ids.

## Setup and training (Laptop 1, macOS)
```bash
cd ~/Downloads/ml_hackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python build_dataset.py      # ~30 s  -> data/*.parquet, results/trip_inventory.csv
python eda.py                # ~30 s  -> reports/figures, reports/maps, results/eda_summary.csv
python train_models.py       # ~6 min -> results/model_comparison.csv, models/*.joblib, figures
python evaluate_hmm.py       # ~10 s  -> models/selected_model.json, results/hmm_vs_classifier.csv
cd ..
```

## Run the API (Laptop 1)
```bash
cd ~/Downloads/ml_hackathon && source .venv/bin/activate
ipconfig getifaddr en0        # note this IP, Laptop 2 needs it
uvicorn api.main:app --host 0.0.0.0 --port 8000
# evaluator trips: PATHSNAP_EVAL_DIR="/path/to/evaluator_trips" uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "00000007"}`.
Every request is logged in this terminal (trip id, accuracy, latency). Reference routes are read only on the
server for scoring and are never sent to the UI.

## Run the UI (Laptop 2)
```bash
cd ml_hackathon/ui
python3 -m http.server 5500
# open http://localhost:5500 , set API URL to http://<LAPTOP1_IP>:8000 , Connect, pick a trip, Run
```
Both laptops must be on the same Wi-Fi / hotspot. If macOS asks to allow incoming connections for Python, allow it.

## Method summary
* **Projection:** each trip uses its own equirectangular projection (metres), since trips are in different regions.
* **Reference labels:** `.route` has no per-point alignment, so each GPS point is assigned to a route arc by a
  forward-only dynamic program minimising point-to-arc distance. Trips 33 and 54 have reference routes ~120–140 m
  from the GPS and are excluded from training.
* **Candidates:** arcs within 60 m, k ≤ 8 nearest.
* **Features:** distance, distance rank, distance minus the closest distance, distance ratio, GPS-heading vs arc-bearing difference,
  heading-valid flag, speed, projection fraction, arc length, #candidates, reverse-twin flag, node degrees.
* **Models:** Logistic Regression, Gaussian NB, KNN, Random Forest, HistGradientBoosting; 5-fold GroupKFold by trip.
* **HMM:** states = candidate arcs; emission = classifier probability; transition = exp(-|d_route − d_gps|/β)
  with bounded Dijkstra; gaps > 30 s split the chain; unreachable pairs get a fixed penalty.

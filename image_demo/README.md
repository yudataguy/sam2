# SAM 2 Image Demo

A lightweight JSON-first demo that focuses on single-image segmentation with SAM 2. The stack consists of:

- **Flask backend** (`image_demo/backend/app.py`) that exposes `/segment-image`. It accepts an uploaded image, runs the SAM 2 automatic mask generator, and returns plain JSON (optionally saving a copy locally).
- **React + Vite frontend** (`image_demo/frontend`) that lets you upload an image, trigger segmentation, visualize colored masks, and download the JSON file for downstream training pipelines.

## Prerequisites

- Python ≥ 3.10 with PyTorch 2.5.1+ installed (see the main `README.md` for installing SAM 2 dependencies: `pip install -e '.[interactive-demo]'`).
- Node.js ≥ 18 for the frontend.
- SAM 2 checkpoints downloaded into `./checkpoints` (run `cd checkpoints && ./download_ckpts.sh`).

## Running the backend

```bash
cd image_demo/backend
APP_ROOT="$(pwd)/../.." \
MODEL_SIZE=base_plus \
PYTORCH_ENABLE_MPS_FALLBACK=1 \  # optional on Apple Silicon
python app.py
```

```bash
cd image_demo/backend
APP_ROOT="$(pwd)/../.." \
MODEL_SIZE=base_plus \
python app.py
```

Environment knobs:

- `MODEL_SIZE`: `tiny`, `small`, `base_plus` (default), or `large`.
- `SAM2_IMAGE_FORCE_CPU=1` forces CPU inference.
- `POINTS_PER_SIDE`, `PRED_IOU_THRESH`, `STABILITY_THRESH` tune the automatic mask generator.
- `SEGMENT_JSON_DIR` changes where saved JSON exports live (defaults to `image_demo/outputs`).
- Requests may override the loaded checkpoint by sending `model_size` (`tiny|small|base_plus|large`). The backend caches each size lazily so you can swap from the UI without restarting the server.

The API now serves two routes:

- `GET /health` – basic status check.
- `POST /segment-image` – multipart form field `image` (required) and optional `max_masks`, `save` flags. Response is pure JSON with image metadata, the colored segments, and an optional `saved_to` relative path if persistence is toggled.

## Running the frontend

```bash
cd image_demo/frontend
npm install
VITE_BACKEND_URL=http://localhost:5050 npm run dev -- --host
```

Open the printed URL (defaults to http://localhost:5174). The UI flow is:

1. Configure the backend URL (defaults to `http://localhost:5050`).
2. Upload an image (PNG/JPEG/etc.).
3. Choose the number of masks, pick the SAM 2 checkpoint (`tiny/small/base_plus/large`), and decide whether the backend should persist JSON.
4. Click **Segment Image** to send a JSON-only request.
5. View colored overlays stacked on top of the original image, inspect per-mask stats, and download the JSON payload for other pipelines.

The downloaded JSON mirrors the backend response, so any downstream training service can consume the same structure without touching GraphQL.

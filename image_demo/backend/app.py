"""Simple Flask backend for image segmentation with SAM 2."""
from __future__ import annotations

import colorsys
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

logger = logging.getLogger(__name__)

APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[2])).resolve()
CHECKPOINTS_DIR = APP_ROOT / "checkpoints"
OUTPUT_DIR = Path(os.environ.get("SEGMENT_JSON_DIR", APP_ROOT / "image_demo" / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_SIZE = os.environ.get("MODEL_SIZE", "base_plus").lower()
MODEL_LOOKUP = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def _resolve_device() -> torch.device:
    force_cpu = os.environ.get("SAM2_IMAGE_FORCE_CPU", "0") == "1"
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not force_cpu:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("using device %s", device)
    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        _enable_tf32()
    elif device.type == "mps":
        logger.warning(
            "MPS support is experimental and may lead to numerical differences vs. CUDA"
        )
    return device


def _enable_tf32() -> None:
    """
    Enable TF32 math on Ampere+ GPUs using the new PyTorch API when available,
    while remaining compatible with earlier releases.
    """
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)

    if hasattr(matmul_backend, "fp32_precision"):
        # New API (PyTorch >= 2.5)
        matmul_backend.fp32_precision = "tf32"
    else:  # pragma: no cover - legacy fall back
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(cudnn_conv, "fp32_precision"):
        cudnn_conv.fp32_precision = "tf32"
    else:  # pragma: no cover - legacy fall back
        torch.backends.cudnn.allow_tf32 = True


def _load_mask_generator(model_size: str) -> SAM2AutomaticMaskGenerator:
    config_name, ckpt_name = MODEL_LOOKUP.get(model_size, MODEL_LOOKUP["base_plus"])
    checkpoint = CHECKPOINTS_DIR / ckpt_name
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} not found. Make sure you've downloaded the SAM 2 weights."
        )

    device = _resolve_device()
    model = build_sam2(config_file=config_name, ckpt_path=str(checkpoint), device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        output_mode="coco_rle",
        points_per_side=int(os.environ.get("POINTS_PER_SIDE", 32)),
        pred_iou_thresh=float(os.environ.get("PRED_IOU_THRESH", 0.86)),
        stability_score_thresh=float(os.environ.get("STABILITY_THRESH", 0.92)),
    )
    return generator


_GENERATOR_CACHE: Dict[str, SAM2AutomaticMaskGenerator] = {}
_CACHE_LOCK = Lock()


def _get_mask_generator(model_size: str | None) -> SAM2AutomaticMaskGenerator:
    resolved = (model_size or DEFAULT_MODEL_SIZE).lower()
    if resolved not in MODEL_LOOKUP:
        raise ValueError(
            f"Unsupported model_size '{model_size}'. Choose from {list(MODEL_LOOKUP.keys())}."
        )

    with _CACHE_LOCK:
        if resolved not in _GENERATOR_CACHE:
            logger.info("loading SAM 2 model %s", resolved)
            _GENERATOR_CACHE[resolved] = _load_mask_generator(resolved)
    return _GENERATOR_CACHE[resolved]

app = Flask(__name__)
CORS(
    app,
    resources={r"/segment-image": {"origins": "*"}, r"/health": {"origins": "*"}},
    supports_credentials=False,
)


@app.get("/health")
def health() -> Any:
    return {
        "status": "ok",
        "model_size": DEFAULT_MODEL_SIZE,
        "loaded_models": list(_GENERATOR_CACHE.keys()),
    }


@app.post("/segment-image")
def segment_image() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' in multipart form"}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as exc:  # pragma: no cover - PIL error path
        logger.exception("failed to read uploaded image")
        return jsonify({"error": f"failed to read image: {exc}"}), 400

    image_np = np.array(image)

    model_size = request.form.get("model_size") or request.args.get("model_size")

    try:
        generator = _get_mask_generator(model_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        masks = generator.generate(image_np)
    except Exception as exc:  # pragma: no cover - model error path
        logger.exception("segmentation failed")
        return jsonify({"error": f"segmentation failed: {exc}"}), 500

    max_masks = request.form.get("max_masks") or request.args.get("max_masks")
    limit = int(max_masks) if max_masks else None

    sorted_masks = sorted(
        masks,
        key=lambda m: float(m.get("predicted_iou", 0.0)),
        reverse=True,
    )
    if limit is not None:
        sorted_masks = sorted_masks[:limit]

    payload_masks = [_serialize_mask(i, mask, image.size) for i, mask in enumerate(sorted_masks)]

    response_body: Dict[str, Any] = {
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "count": len(payload_masks),
        "model_size": (model_size or DEFAULT_MODEL_SIZE).lower(),
        "segments": payload_masks,
    }

    save_flag = request.form.get("save", request.args.get("save"))
    if save_flag and save_flag not in {"0", "false", "False", None}:
        filename = _write_output(response_body)
        response_body["saved_to"] = filename

    return jsonify(response_body)


def _to_list(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if hasattr(value, "tolist"):
        raw_list = value.tolist()
        if isinstance(raw_list, list):
            return [float(v) for v in raw_list]
    return []


def _serialize_mask(idx: int, mask: Dict[str, Any], image_size: Any) -> Dict[str, Any]:
    segmentation = mask.get("segmentation", {})
    if isinstance(segmentation, list):
        # Already polygon data
        encoded = segmentation
    else:
        counts = segmentation.get("counts") if isinstance(segmentation, dict) else None
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        encoded = {
            "counts": counts,
            "size": segmentation.get("size", [image_size[1], image_size[0]]),
        }

    return {
        "id": idx,
        "bbox": _to_list(mask.get("bbox", [])),
        "area": float(mask.get("area", 0.0)),
        "predicted_iou": float(mask.get("predicted_iou", 0.0)),
        "stability_score": float(mask.get("stability_score", 0.0)),
        "crop_box": _to_list(mask.get("crop_box", [])),
        "point_coords": [
            _to_list(point) for point in mask.get("point_coords", [])
        ],
        "segmentation": encoded,
        "color": _index_to_hex(idx),
    }


def _index_to_hex(idx: int) -> str:
    hue = (idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _write_output(payload: Dict[str, Any]) -> str:
    timestamp = int(time.time() * 1000)
    filename = OUTPUT_DIR / f"segmentation_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return str(filename.relative_to(APP_ROOT))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)

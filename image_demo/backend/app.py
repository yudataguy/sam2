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
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

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
_PREDICTOR_CACHE: Dict[str, SAM2ImagePredictor] = {}
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


def _load_image_predictor(model_size: str) -> SAM2ImagePredictor:
    config_name, ckpt_name = MODEL_LOOKUP.get(model_size, MODEL_LOOKUP["base_plus"])
    checkpoint = CHECKPOINTS_DIR / ckpt_name
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} not found. Make sure you've downloaded the SAM 2 weights."
        )

    device = _resolve_device()
    model = build_sam2(config_file=config_name, ckpt_path=str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def _get_image_predictor(model_size: str | None) -> SAM2ImagePredictor:
    resolved = (model_size or DEFAULT_MODEL_SIZE).lower()
    if resolved not in MODEL_LOOKUP:
        raise ValueError(
            f"Unsupported model_size '{model_size}'. Choose from {list(MODEL_LOOKUP.keys())}."
        )

    with _CACHE_LOCK:
        if resolved not in _PREDICTOR_CACHE:
            logger.info("loading SAM 2 image predictor %s", resolved)
            _PREDICTOR_CACHE[resolved] = _load_image_predictor(resolved)
    return _PREDICTOR_CACHE[resolved]

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/segment-image": {"origins": "*"},
        r"/refine-segment": {"origins": "*"},
        r"/create-mask": {"origins": "*"},
        r"/save-project": {"origins": "*"},
        r"/health": {"origins": "*"}
    },
    supports_credentials=False,
)


def _decode_rle_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Decode COCO RLE to binary mask array."""
    if mask_utils is None:
        raise RuntimeError("pycocotools not available for mask decoding")
    return mask_utils.decode(rle)


def _encode_rle_mask(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Encode binary mask array to COCO RLE format."""
    if mask_utils is None:
        raise RuntimeError("pycocotools not available for mask encoding")
    # Ensure mask is in Fortran order (column-major) as expected by pycocotools
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    # Decode bytes to string
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _subtract_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Subtract mask_b from mask_a (keeps parts of A not in B)."""
    return np.logical_and(mask_a, np.logical_not(mask_b)).astype(np.uint8)


def _compute_mask_metrics(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Compute area and bounding box for a binary mask."""
    area = int(np.sum(binary_mask))
    if area == 0:
        return {"area": 0, "bbox": [0, 0, 0, 0]}

    # Find bounding box
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # COCO bbox format: [x, y, width, height]
    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    return {"area": area, "bbox": bbox}


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


@app.post("/refine-segment")
def refine_segment() -> Any:
    """Refine a segment by re-segmenting within a bounding box."""
    # Parse request
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' in multipart form"}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as exc:
        logger.exception("failed to read uploaded image")
        return jsonify({"error": f"failed to read image: {exc}"}), 400

    image_np = np.array(image)

    # Get parameters
    try:
        bbox_str = request.form.get("bbox")
        if not bbox_str:
            return jsonify({"error": "missing 'bbox' parameter"}), 400
        bbox = json.loads(bbox_str)  # [x, y, width, height]

        segment_id_str = request.form.get("segment_id")
        if segment_id_str is None:
            return jsonify({"error": "missing 'segment_id' parameter"}), 400
        segment_id = int(segment_id_str)

        segments_str = request.form.get("segments")
        if not segments_str:
            return jsonify({"error": "missing 'segments' parameter"}), 400
        segments = json.loads(segments_str)

    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"error": f"invalid parameter format: {exc}"}), 400

    # Find the segment to refine
    target_segment = None
    for seg in segments:
        if seg.get("id") == segment_id:
            target_segment = seg
            break

    if target_segment is None:
        return jsonify({"error": f"segment with id {segment_id} not found"}), 404

    # Get predictor
    model_size = request.form.get("model_size") or request.args.get("model_size")
    try:
        predictor = _get_image_predictor(model_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Set image for predictor
    try:
        predictor.set_image(image_np)
    except Exception as exc:
        logger.exception("failed to set image")
        return jsonify({"error": f"failed to set image: {exc}"}), 500

    # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
    x, y, w, h = bbox
    input_box = np.array([x, y, x + w, y + h])

    # Predict with bbox prompt
    try:
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=False,  # Single best mask
        )
    except Exception as exc:
        logger.exception("refinement prediction failed")
        return jsonify({"error": f"prediction failed: {exc}"}), 500

    # Get the refined mask (first mask since multimask_output=False)
    refined_mask = masks[0]  # Shape: (H, W), boolean

    # Decode original segment mask
    original_rle = target_segment.get("segmentation", {})
    try:
        original_mask = _decode_rle_mask(original_rle)
    except Exception as exc:
        logger.exception("failed to decode original mask")
        return jsonify({"error": f"failed to decode original mask: {exc}"}), 500

    # Split the mask: refined part and remainder
    # The refined_mask is the new segment inside the bbox
    # The remainder is original_mask minus refined_mask
    remainder_mask = _subtract_masks(original_mask, refined_mask)

    # Encode masks to RLE
    try:
        refined_rle = _encode_rle_mask(refined_mask.astype(np.uint8))
        remainder_rle = _encode_rle_mask(remainder_mask)
    except Exception as exc:
        logger.exception("failed to encode masks")
        return jsonify({"error": f"failed to encode masks: {exc}"}), 500

    # Compute metrics for new masks
    refined_metrics = _compute_mask_metrics(refined_mask.astype(np.uint8))
    remainder_metrics = _compute_mask_metrics(remainder_mask)

    # Generate unique IDs (use timestamp-based approach)
    timestamp = int(time.time() * 1000)
    refined_id = f"refined_{timestamp}"
    remainder_id = f"remainder_{timestamp}"

    # Create new segment objects
    refined_segment = {
        "id": refined_id,
        "bbox": refined_metrics["bbox"],
        "area": refined_metrics["area"],
        "predicted_iou": float(scores[0]),
        "stability_score": 0.0,  # Not available from predictor
        "crop_box": bbox,  # Store the refinement bbox
        "point_coords": [],
        "segmentation": refined_rle,
        "color": target_segment.get("color", "#000000"),  # Keep original color initially
        "refined_from": segment_id,  # Track origin
    }

    # Only return remainder if it has non-zero area
    updated_segments = []
    if remainder_metrics["area"] > 0:
        remainder_segment = {
            "id": remainder_id,
            "bbox": remainder_metrics["bbox"],
            "area": remainder_metrics["area"],
            "predicted_iou": target_segment.get("predicted_iou", 0.0),
            "stability_score": target_segment.get("stability_score", 0.0),
            "crop_box": target_segment.get("crop_box", []),
            "point_coords": target_segment.get("point_coords", []),
            "segmentation": remainder_rle,
            "color": target_segment.get("color", "#000000"),
            "remainder_of": segment_id,  # Track origin
        }
        updated_segments.append(remainder_segment)

    updated_segments.append(refined_segment)

    return jsonify({
        "refined_segment": refined_segment,
        "remainder_segment": updated_segments[0] if len(updated_segments) > 1 else None,
        "original_segment_id": segment_id,
    })


@app.post("/create-mask")
def create_mask() -> Any:
    """Create a new independent mask from a bounding box region."""
    # Parse request
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' in multipart form"}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as exc:
        logger.exception("failed to read uploaded image")
        return jsonify({"error": f"failed to read image: {exc}"}), 400

    image_np = np.array(image)

    # Get parameters
    try:
        bbox_str = request.form.get("bbox")
        if not bbox_str:
            return jsonify({"error": "missing 'bbox' parameter"}), 400
        bbox = json.loads(bbox_str)  # [x, y, width, height]

    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"error": f"invalid parameter format: {exc}"}), 400

    # Get predictor
    model_size = request.form.get("model_size") or request.args.get("model_size")
    try:
        predictor = _get_image_predictor(model_size)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Set image for predictor
    try:
        predictor.set_image(image_np)
    except Exception as exc:
        logger.exception("failed to set image")
        return jsonify({"error": f"failed to set image: {exc}"}), 500

    # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
    x, y, w, h = bbox
    input_box = np.array([x, y, x + w, y + h])

    # Predict with bbox prompt
    try:
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True,  # Get multiple mask candidates
        )
    except Exception as exc:
        logger.exception("mask creation prediction failed")
        return jsonify({"error": f"prediction failed: {exc}"}), 500

    # Get all masks and scores
    # masks shape: (num_masks, H, W)
    payload_masks = []

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Encode mask to RLE
        try:
            mask_rle = _encode_rle_mask(mask.astype(np.uint8))
        except Exception as exc:
            logger.exception("failed to encode mask")
            continue

        # Compute metrics
        metrics = _compute_mask_metrics(mask.astype(np.uint8))

        # Create segment object
        segment = {
            "id": None,  # Will be assigned by frontend
            "bbox": metrics["bbox"],
            "area": metrics["area"],
            "predicted_iou": float(score),
            "stability_score": 0.0,  # Not available from predictor
            "segmentation": mask_rle,
            "color": _index_to_hex(i),  # Generate color for this mask
        }
        payload_masks.append(segment)

    if not payload_masks:
        return jsonify({"error": "no masks generated from the bounding box"}), 400

    # Sort by area/score and return best candidate(s)
    sorted_masks = sorted(
        payload_masks,
        key=lambda m: float(m.get("predicted_iou", 0.0)),
        reverse=True,
    )

    return jsonify({
        "masks": sorted_masks,
        "bbox": bbox,
    })


@app.post("/save-project")
def save_project() -> Any:
    """Save project in COCO JSON format."""
    try:
        # Parse request data
        data_str = request.form.get("data")
        if not data_str:
            return jsonify({"error": "missing 'data' parameter"}), 400

        data = json.loads(data_str)
        image_metadata = data.get("image", {})
        segments = data.get("segments", [])
        labels_list = data.get("labels", [])

    except (json.JSONDecodeError, KeyError) as exc:
        return jsonify({"error": f"invalid request format: {exc}"}), 400

    # Build COCO format
    # Create categories from labels
    categories = []
    label_to_id = {}
    for idx, label in enumerate(labels_list, start=1):
        categories.append({
            "id": idx,
            "name": label,
            "supercategory": "object"
        })
        label_to_id[label] = idx

    # Create image entry
    timestamp = int(time.time() * 1000)
    image_filename = data.get("image_filename", f"image_{timestamp}.jpg")
    coco_image = {
        "id": 1,
        "file_name": image_filename,
        "width": image_metadata.get("width", 0),
        "height": image_metadata.get("height", 0),
    }

    # Create annotations from segments
    annotations = []
    for seg in segments:
        # Only include segments with labels
        label = seg.get("label")
        if not label or label not in label_to_id:
            continue

        annotation = {
            "id": len(annotations) + 1,
            "image_id": 1,
            "category_id": label_to_id[label],
            "bbox": seg.get("bbox", []),
            "area": float(seg.get("area", 0)),
            "segmentation": seg.get("segmentation", {}),
            "iscrowd": 0,
            "score": float(seg.get("predicted_iou", 0.0)),
        }
        annotations.append(annotation)

    coco_format = {
        "images": [coco_image],
        "annotations": annotations,
        "categories": categories,
    }

    # Save to file
    output_dir = OUTPUT_DIR / "coco_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"coco_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, indent=2)

    relative_path = str(filename.relative_to(APP_ROOT))

    return jsonify({
        "status": "success",
        "saved_to": relative_path,
        "annotations_count": len(annotations),
        "categories_count": len(categories),
    })


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

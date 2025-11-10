# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAM 2 (Segment Anything Model 2) is a foundation model for promptable visual segmentation in images and videos. It extends the original SAM model to handle video by treating images as single-frame videos, using a streaming memory architecture for real-time video processing.

**Key capabilities:**
- Promptable segmentation on images (points, boxes, masks)
- Multi-object tracking in videos with temporal propagation
- Automatic mask generation
- Training/fine-tuning on custom datasets

## Installation and Setup

```bash
# Basic installation (requires Python >=3.10, PyTorch >=2.5.1, torchvision >=0.20.1)
pip install -e .

# With notebook dependencies (required for examples)
pip install -e ".[notebooks]"

# With development/training dependencies
pip install -e ".[dev]"

# Download model checkpoints (required before first use)
cd checkpoints && ./download_ckpts.sh && cd ..
```

**Note:** Installation compiles a custom CUDA kernel (`sam2._C`). If compilation fails, SAM 2 still works but some post-processing may be limited.

## Common Commands

### Running Examples
```bash
# Launch Jupyter notebooks (see notebooks/ directory)
jupyter notebook notebooks/image_predictor_example.ipynb
jupyter notebook notebooks/video_predictor_example.ipynb
jupyter notebook notebooks/automatic_mask_generator_example.ipynb

# Benchmark video inference performance
python sam2/benchmark.py

# Run web demo (requires Docker)
docker compose up --build
```

### Training/Fine-tuning
```bash
# Fine-tune on a dataset (example: MOSE with 8 GPUs)
python training/train.py \
    -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
    --use-cluster 0 \
    --num-gpus 8

# Multi-node training with SLURM
python training/train.py \
    -c <config.yaml> \
    --use-cluster 1 \
    --num-gpus 8 \
    --num-nodes 2 \
    --partition $PARTITION
```

Logs and checkpoints save to `sam2_logs/` by default. Monitor training with TensorBoard logs in `<log_dir>/tensorboard/`.

### Video Object Segmentation (VOS) Evaluation
```bash
# Generate predictions on a video dataset
python tools/vos_inference.py \
    --config <model_config.yaml> \
    --checkpoint <checkpoint.pt> \
    --dataset_dir <path_to_dataset> \
    --output_dir <output_path>

# Evaluate predictions
python sav_dataset/sav_evaluator.py \
    --gt_root <ground_truth_dir> \
    --pred_root <predictions_dir>
```

## Architecture Overview

### Core Model Components (sam2/modeling/)

**SAM2Base** (`sam2_base.py`, ~47k LOC)
- Main model combining all components
- Handles both image-only and video inference
- Manages streaming memory for temporal context

**Image Encoder** (`backbones/image_encoder.py`)
- Hierarchical Vision Transformer (Hiera backbone)
- Extracts multi-scale image features
- 4 model sizes: tiny (38.9M), small (46M), base+ (80.8M), large (224.4M)

**Memory Encoder & Attention** (`memory_encoder.py`, `memory_attention.py`)
- Encodes masks and image features into memory
- Cross-attends between current frame and memory bank
- Enables temporal propagation in videos

**Mask Decoder** (`sam/mask_decoder.py`)
- Transformer-based decoder from original SAM
- Takes prompts (points/boxes) + image embeddings → masks
- Predicts masks and IoU scores

**Position Encoding** (`position_encoding.py`)
- RoPE (Rotary Position Embedding) for spatial encoding
- Temporal position encoding for video frames

### Inference APIs (sam2/)

**build_sam.py**
- Factory functions: `build_sam2()`, `build_sam2_video_predictor()`
- Loads configs from `configs/` and checkpoints
- Set `vos_optimized=True` for major speedup via `torch.compile`

**SAM2ImagePredictor** (`sam2_image_predictor.py`)
- API for static images (similar to original SAM)
- Methods: `set_image()`, `predict()`, `predict_torch()`
- Supports iterative refinement with multiple prompts

**SAM2VideoPredictor** (`sam2_video_predictor.py`)
- Stateful video API with independent per-object tracking
- Core workflow:
  1. `init_state(video_path)` - Initialize inference state
  2. `add_new_points_or_box(state, frame_idx, obj_id, ...)` - Add prompts
  3. `propagate_in_video(state)` - Propagate masks through time
- Supports adding new objects mid-tracking (as of Dec 2024 update)

**AutomaticMaskGenerator** (`automatic_mask_generator.py`)
- Generates masks over entire image without prompts
- Grid-based sampling + NMS post-processing
- Compatible with original SAM's automatic mode

### Training Infrastructure (training/)

**Trainer** (`trainer.py`)
- Main train/eval loop
- Handles distributed training (DDP)
- Integrates with Hydra for config management

**Model** (`model/sam2_train.py`)
- `SAM2Train` wraps `SAM2Base` for training
- Simulates user prompts (iterative point sampling)
- Supports mixed image + video training

**Datasets** (`dataset/`)
- `VOSDataset` - Video object segmentation datasets
- `SA1BRawDataset` - SA-1B image dataset
- `JSONRawDataset` - SA-V video dataset
- `DAVISRawDataset` - DAVIS-style datasets (MOSE, etc.)
- `TorchTrainMixedDataset` - Mix multiple datasets

**Loss Functions** (`loss_fns.py`)
- `MultiStepMultiMasksAndIous` - Main training loss
- Combines mask prediction + IoU prediction losses
- Supports multi-step refinement simulation

**Config System**
- Hydra-based configuration in `configs/`
- Model configs: `configs/sam2.1/*.yaml` (current), `configs/sam2/*.yaml` (legacy)
- Training configs: `configs/sam2.1_training/*.yaml`

## Key Design Patterns

### 1. Inference State Management (Videos)
Video prediction uses stateful tracking via `inference_state` dictionary:
```python
state = predictor.init_state(video_path)  # Initialize
predictor.add_new_points_or_box(state, ...)  # Add prompts
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    # Process masks
```
State contains memory bank, object embeddings, and tracking history.

### 2. Config-Driven Model Building
Models are built via YAML configs + factory functions:
```python
from sam2.build_sam import build_sam2
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
```
Configs specify: backbone, memory encoder, mask decoder, image size, etc.

### 3. Streaming Memory
Video frames processed sequentially, maintaining fixed-size memory bank:
- Recent frames always in memory
- Older frames summarized via memory encoder
- Enables real-time processing without loading entire video

### 4. Prompt Simulation (Training)
Training simulates interactive prompting:
- Sample positive/negative points on ground truth masks
- Iterative refinement (multiple rounds)
- Random temporal sampling for videos

## Model Checkpoints

**SAM 2.1 (Released Sep 2024)** - Current, improved checkpoints:
- `sam2.1_hiera_tiny.pt` - 38.9M params, 91.2 FPS
- `sam2.1_hiera_small.pt` - 46M params, 84.8 FPS
- `sam2.1_hiera_base_plus.pt` - 80.8M params, 64.1 FPS
- `sam2.1_hiera_large.pt` - 224.4M params, 39.5 FPS

Checkpoints can be loaded from local files or Hugging Face Hub:
```python
# Local
predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)

# Hugging Face
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
```

## Important Constants/Defaults

- **Image size:** 1024x1024 (default), configurable in model configs
- **Video memory:** Configurable number of frames in memory bank
- **Precision:** `torch.bfloat16` recommended for inference (A100+ GPUs)
- **Compilation:** Set `vos_optimized=True` for ~2-3x speedup on video inference

## Project Structure

```
sam2/
├── modeling/              # Core neural network components
│   ├── sam2_base.py      # Main SAM2 model
│   ├── backbones/        # Image encoders (Hiera)
│   └── sam/              # Mask decoder (from original SAM)
├── build_sam.py          # Model factory functions
├── sam2_image_predictor.py
├── sam2_video_predictor.py
├── automatic_mask_generator.py
├── configs/              # Model configuration files
└── csrc/                 # CUDA extension (connected components)

training/
├── train.py              # Training entry point
├── trainer.py            # Main training loop
├── model/                # SAM2Train wrapper
├── dataset/              # Dataloaders (SA-1B, SA-V, MOSE, DAVIS)
└── utils/                # Training utilities

notebooks/                # Example Jupyter notebooks
demo/                     # Web demo (React + Flask + GraphQL)
tools/                    # VOS inference scripts
sav_dataset/              # SA-V dataset tools and evaluation
```

## Notes

- **CUDA requirement:** Custom kernel requires nvcc compiler (matches PyTorch CUDA version)
- **Memory usage:** Large model + long videos require significant GPU memory (80GB for training)
- **Compilation:** First-time model loading triggers JIT compilation if using `torch.compile`
- **Frame naming:** Video frames should be JPEG with numeric names (e.g., `00000.jpg`, `00001.jpg`)

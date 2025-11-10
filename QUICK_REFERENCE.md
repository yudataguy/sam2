# SAM 2 Codebase - Quick Reference Guide

## Most Important Files

### Entry Points
- `/sam2/build_sam.py` - Model factory functions (start here!)
- `/sam2/sam2_image_predictor.py` - Image segmentation API
- `/sam2/sam2_video_predictor.py` - Video tracking API
- `/training/train.py` - Training launcher

### Core Architecture
- `/sam2/modeling/sam2_base.py` - Main SAM2 model (909 lines)
- `/sam2/modeling/memory_encoder.py` - Video memory handling
- `/sam2/modeling/memory_attention.py` - Temporal attention
- `/sam2/modeling/backbones/hieradet.py` - Image encoder

### Demos & Examples
- `/notebooks/image_predictor_example.ipynb` - Image segmentation example
- `/notebooks/video_predictor_example.ipynb` - Video tracking example
- `/image_demo/backend/app.py` - Simple JSON API demo
- `/demo/backend/server/app.py` - Full GraphQL web demo

---

## File Organization by Use Case

### I want to segment static images
1. `sam2/build_sam.py` - Load model with `build_sam2()`
2. `sam2/sam2_image_predictor.py` - Use `SAM2ImagePredictor` class
3. Reference: `notebooks/image_predictor_example.ipynb`

### I want to track objects in video
1. `sam2/build_sam.py` - Load predictor with `build_sam2_video_predictor()`
2. `sam2/sam2_video_predictor.py` - Use `SAM2VideoPredictor` class
3. Reference: `notebooks/video_predictor_example.ipynb`

### I want to generate masks automatically (no prompts)
1. `sam2/automatic_mask_generator.py` - Use `SAM2AutomaticMaskGenerator`
2. Reference: `notebooks/automatic_mask_generator_example.ipynb`

### I want to fine-tune the model
1. `training/train.py` - Launch training script
2. `training/trainer.py` - Main training loop
3. Config: `training/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml`
4. Reference: `training/README.md`

### I want to deploy as a web service
- Full demo: `/demo/` (Flask + GraphQL + React)
- Simple demo: `/image_demo/` (Flask + JSON + React)
- Both have backend and frontend subdirectories

### I want to evaluate on a dataset
1. `tools/vos_inference.py` - Generate predictions
2. `sav_dataset/sav_evaluator.py` - Evaluate metrics

---

## Key Classes

### Model Building
```python
from sam2.build_sam import build_sam2, build_sam2_video_predictor

# Image model
model = build_sam2(
    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
    ckpt_path="./checkpoints/sam2.1_hiera_large.pt"
)

# Video predictor
predictor = build_sam2_video_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
    ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
    vos_optimized=True  # torch.compile optimization
)
```

### Image Prediction
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(model)
predictor.set_image(image)
masks, iou, _ = predictor.predict(
    point_coords=[[x, y]],
    point_labels=[1]
)
```

### Video Tracking
```python
from sam2.sam2_video_predictor import SAM2VideoPredictor

state = predictor.init_state(video_path)

# Add prompt on frame 0
frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    state, 
    frame_idx=0,
    obj_id=0,
    points=np.array([[x, y]])
)

# Track through video
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    process_frame(masks)
```

### Automatic Masks
```python
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

mask_generator = SAM2AutomaticMaskGenerator(model)
masks = mask_generator.generate(image)

# Returns list of dicts with keys: segmentation, bbox, area, predicted_iou, etc.
for mask_dict in masks:
    mask = mask_dict['segmentation']
    ...
```

---

## Model Sizes & Performance

| Model | Size | Speed | SA-V J&F | MOSE J&F | LVOS J&F |
|-------|------|-------|----------|----------|----------|
| tiny | 38.9M | 91.2 FPS | 76.5 | 71.8 | 77.3 |
| small | 46M | 84.8 FPS | 76.6 | 73.5 | 78.3 |
| base+ | 80.8M | 64.1 FPS | 78.2 | 73.7 | 78.2 |
| large | 224.4M | 39.5 FPS | 79.5 | 74.6 | 80.6 |

*SAM 2.1 metrics; speed on A100 with torch 2.5.1*

---

## Configuration Files

### Model Configs
- Original: `sam2/configs/sam2/{tiny,small,b+,large}.yaml`
- Improved: `sam2/configs/sam2.1/{tiny,small,b+,large}.yaml`

### Training Config Example
```yaml
model:
  _target_: training.model.sam2.SAM2Train
  image_encoder: ...
  memory_attention: ...
  
dataset:
  img_folder: null  # Set to dataset path
  gt_folder: null
  
optimizer:
  lr: 1e-4
```

---

## Common Patterns

### Loading from Hugging Face
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

from sam2.sam2_video_predictor import SAM2VideoPredictor
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
```

### Batch Image Processing
```python
predictor = SAM2ImagePredictor(model)

for img in image_list:
    predictor.set_image(img)
    # Make predictions...
    predictor.reset_image()
```

### Device Management
```python
# GPU with bfloat16 (recommended)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(...)

# CPU or MPS
model = build_sam2(..., device="cpu")
```

### Memory Optimization
```python
# For video prediction
state = predictor.init_state(
    video_path,
    offload_video_to_cpu=True,      # Save GPU memory
    offload_state_to_cpu=False      # Keep inference fast
)
```

---

## Dependencies by Use Case

### Minimal (inference only)
```bash
pip install torch torchvision numpy hydra-core pillow tqdm iopath
```

### Notebooks
```bash
pip install -e ".[notebooks]"
```

### Web Demo
```bash
pip install -e ".[interactive-demo]"
```

### Training
```bash
pip install -e ".[dev]"
```

---

## Directory Tree (Key Files Only)

```
sam2/
├── sam2/
│   ├── build_sam.py              # START HERE
│   ├── sam2_image_predictor.py    # Image API
│   ├── sam2_video_predictor.py    # Video API
│   ├── automatic_mask_generator.py # Auto masks
│   ├── modeling/
│   │   ├── sam2_base.py           # Core model
│   │   ├── memory_*.py            # Video components
│   │   ├── backbones/hieradet.py  # Image encoder
│   │   └── sam/                   # Original SAM
│   └── configs/
│       ├── sam2/                  # Original checkpoints
│       └── sam2.1/                # Improved checkpoints
│
├── training/
│   ├── train.py                   # Training launcher
│   ├── trainer.py                 # Main loop
│   ├── dataset/                   # Data loading
│   └── configs/                   # Training configs
│
├── demo/                          # Full web demo
├── image_demo/                    # Simple JSON demo
├── notebooks/                     # Examples
├── tools/                         # Evaluation
└── sav_dataset/                   # Dataset utils
```

---

## Debugging Tips

### Model won't load
- Check checkpoint path exists
- Verify config matches checkpoint (e.g., sam2.1 config + sam2 checkpoint mismatch)
- Check CUDA/device availability

### Slow performance
- Use smaller model size (tiny/small vs large)
- Enable torch.compile: `vos_optimized=True`
- Use bfloat16: `torch.autocast("cuda", dtype=torch.bfloat16)`
- Offload frames to CPU: `offload_video_to_cpu=True`

### Memory errors
- Reduce image resolution (though 512x512 is standard)
- Reduce video frames in batch
- Enable state offloading: `offload_state_to_cpu=True`
- Use smaller model

### Poor mask quality
- Try different model size (larger is better)
- Adjust automatic mask generator thresholds:
  - `pred_iou_thresh` (default 0.86)
  - `stability_score_thresh` (default 0.92)
- For video, ensure good initial prompt

---

## Important Constants & Defaults

```python
# Image processing
DEFAULT_IMAGE_SIZE = 512
BACKBONE_STRIDE = 16

# Video processing
DEFAULT_NUM_MEMORY_FRAMES = 7      # num_maskmem
DEFAULT_MAX_OBJ_PTRS = 16          # max tracked objects

# Mask generator
DEFAULT_POINTS_PER_SIDE = 32
DEFAULT_PRED_IOU_THRESH = 0.86
DEFAULT_STABILITY_THRESH = 0.92

# Training
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-4
DEFAULT_NUM_EPOCHS = 50
```

---

## Version Notes

- **Current:** SAM 2.1 (Sept 2024) - Use this
- **Previous:** SAM 2 (July 2024) - Legacy support maintained
- **Oldest:** SAM 1.0 - Different codebase

---

## Where to Find Things

- **Installation issues?** → `INSTALL.md`
- **Need examples?** → `notebooks/`
- **Training docs?** → `training/README.md`
- **Web demo setup?** → `demo/README.md`
- **Dataset info?** → `sav_dataset/README.md`
- **Latest changes?** → `RELEASE_NOTES.md`

---

**Last Updated:** November 2024
**Codebase Location:** `/home/sam/Downloads/code/playground/sam2/`
**Full Documentation:** See `CODEBASE_OVERVIEW.md` for comprehensive details

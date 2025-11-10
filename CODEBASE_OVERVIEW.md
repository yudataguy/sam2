# SAM 2: Segment Anything in Images and Videos - Codebase Overview

**Project:** SAM 2 - Foundation model for promptable visual segmentation in images and videos  
**Version:** 1.0  
**Authors:** Meta AI (segment-anything@meta.com)  
**Repository:** https://github.com/facebookresearch/sam2  
**License:** Apache 2.0  

---

## Executive Summary

SAM 2 is a comprehensive machine learning framework for image and video segmentation. It extends the original SAM (Segment Anything Model) to handle video sequences with temporal memory, streaming capabilities, and interactive refinement. The codebase is well-organized with clear separation between core modeling, inference APIs, training infrastructure, and demonstration applications.

---

## 1. Main Directory Structure

```
sam2/
├── sam2/                          # Core Python package
│   ├── modeling/                  # Neural network architecture components
│   ├── utils/                     # Utility functions
│   ├── configs/                   # YAML configuration files
│   ├── csrc/                      # CUDA source code (C++)
│   ├── build_sam.py              # Model builder and factory functions
│   ├── automatic_mask_generator.py # Automated segmentation
│   ├── sam2_image_predictor.py    # Image inference API
│   ├── sam2_video_predictor.py    # Video inference API (current)
│   ├── sam2_video_predictor_legacy.py # Legacy video API
│   └── benchmark.py              # Performance benchmarking
│
├── training/                      # Training and fine-tuning code
│   ├── dataset/                   # Data loading and preprocessing
│   ├── model/                     # Training-specific model implementations
│   ├── utils/                     # Training utilities
│   ├── scripts/                   # Dataset processing scripts
│   ├── train.py                   # Training launch script
│   ├── trainer.py                 # Main training loop
│   ├── loss_fns.py               # Loss functions
│   └── optimizer.py              # Optimizer utilities
│
├── demo/                          # Full-stack web demo
│   ├── backend/                   # Flask GraphQL backend
│   ├── frontend/                  # React + Vite frontend
│   └── data/                      # Sample demo data
│
├── image_demo/                    # Lightweight image segmentation demo
│   ├── backend/                   # Flask JSON-based backend
│   ├── frontend/                  # React + Vite UI
│   └── outputs/                   # Generated outputs
│
├── notebooks/                     # Jupyter notebooks for examples
│   ├── images/                    # Sample images
│   ├── videos/                    # Sample videos
│   ├── image_predictor_example.ipynb
│   ├── video_predictor_example.ipynb
│   └── automatic_mask_generator_example.ipynb
│
├── sav_dataset/                   # Segment Anything Video (SA-V) dataset utilities
│   ├── utils/                     # Dataset evaluation and utilities
│   ├── example/                   # Example code
│   ├── sav_evaluator.py          # Evaluation script
│   └── requirements.txt          # Dataset-specific dependencies
│
├── tools/                         # Inference and utility tools
│   └── vos_inference.py          # Video object segmentation inference
│
├── checkpoints/                   # Model weights storage (downloaded separately)
├── assets/                        # Images and resources
│
├── setup.py                       # Package installation configuration
├── pyproject.toml                # Build system configuration
├── docker-compose.yaml           # Docker orchestration
├── backend.Dockerfile            # Backend container definition
├── MANIFEST.in                   # Package data manifest
├── .clang-format                 # Code formatting rules
├── .gitignore                    # Git ignore patterns
│
├── README.md                      # Main documentation
├── INSTALL.md                     # Installation guide
├── CONTRIBUTING.md               # Contribution guidelines
├── CODE_OF_CONDUCT.md            # Code of conduct
├── RELEASE_NOTES.md              # Version history
└── LICENSE                        # Apache 2.0 license
```

---

## 2. Key Python Modules and Responsibilities

### 2.1 Core Module: `sam2/`

#### **`sam2/__init__.py`**
- Initializes Hydra configuration system
- Sets up configuration module discovery
- Manages global Hydra state for configuration composition

#### **`sam2/build_sam.py`** (174 lines)
**Purpose:** Model factory and configuration management

**Key Functions:**
- `build_sam2()` - Builds the SAM2 model for image/inference
- `build_sam2_video_predictor()` - Builds video-specific predictor with memory optimization
- `build_sam2_hf()` - Loads SAM2 models from Hugging Face Hub
- `build_sam2_video_predictor_hf()` - Loads video predictor from HF
- `_load_checkpoint()` - Checkpoint loading and validation
- `_hf_download()` - Downloads models from Hugging Face

**Config Maps:**
- Supports SAM 2 original (July 2024) and SAM 2.1 (September 2024) checkpoints
- Maps model IDs to config paths and checkpoint filenames
- 4 model sizes: tiny (38.9M), small (46M), base_plus (80.8M), large (224.4M)

---

### 2.2 Inference APIs

#### **`sam2/sam2_image_predictor.py`** (466 lines)
**Purpose:** Interactive segmentation on static images

**Main Class: `SAM2ImagePredictor`**
- Single-image mask prediction with prompt-based interaction
- Features:
  - Sets image embedding once, enables multiple efficient predictions
  - Supports point and box prompts
  - Mask refinement through iterative prompts
  - Threshold-based post-processing (hole filling, sprinkle removal)
  - Compatible with both SAM and SAM2 models
  - Batch image support for efficiency
  - `from_pretrained()` class method for HF integration

**Key Methods:**
- `set_image()` - Pre-compute image embeddings
- `predict()` - Generate masks from prompts
- `postprocess_masks()` - Apply threshold and morphological operations
- `reset_image()` - Clear state for new images

---

#### **`sam2/sam2_video_predictor.py`** (1223 lines)
**Purpose:** Temporal segmentation and object tracking in videos

**Main Class: `SAM2VideoPredictor` (extends `SAM2Base`)**
- Multi-frame video tracking with memory-augmented inference
- Features:
  - Streaming memory for real-time processing
  - Multi-object tracking with independent object prompts
  - Dense object propagation through video frames
  - Memory compression (temporal stride support)
  - Inference state management
  - Supports frame-by-frame or full-video processing

**Key Methods:**
- `init_state()` - Initialize inference state for a video
- `add_new_points_or_box()` - Add interaction prompts on specific frame
- `propagate_in_video()` - Propagate masks through video frames
- `reset_state()` - Clear tracking state
- `_track_in_video()` - Core tracking loop with memory management

**Subclass: `SAM2VideoPredictorVOS`**
- Optimized for Video Object Segmentation (VOS) tasks
- Supports torch.compile for major speedup
- Per-object independent inference

---

#### **`sam2/sam2_video_predictor_legacy.py`** (1172 lines)
- Legacy implementation maintained for backward compatibility
- Used by training code and earlier workflows
- Parallel API to current video predictor

---

### 2.3 Automatic Segmentation

#### **`sam2/automatic_mask_generator.py`** (454 lines)
**Purpose:** Unsupervised, dense mask generation on images

**Main Class: `SAM2AutomaticMaskGenerator`**
- Generates comprehensive mask proposals without explicit prompts
- Features:
  - Grid-based point prompt generation
  - Hierarchical mask filtering by quality metrics
  - Multi-scale processing with crops
  - Post-processing for clean mask boundaries
  - Stability and IoU-based filtering
  - Output in multiple formats (RLE, binary)
  - `from_pretrained()` for HF models

**Key Methods:**
- `generate()` - Generate masks for an image
- `_generate_masks()` - Core generation pipeline
- `_process_crop()` - Single crop processing
- `_process_batch()` - Batch point processing
- `postprocess_small_regions()` - Remove small spurious regions
- `refine_with_m2m()` - Multi-mask refinement

---

### 2.4 Modeling Architecture: `sam2/modeling/`

#### **Directory Structure**
```
sam2/modeling/
├── sam2_base.py              # Core SAM2Base class (909 lines)
├── sam2_utils.py             # Utility functions (323 lines)
├── memory_encoder.py         # Video memory processing (181 lines)
├── memory_attention.py       # Temporal attention (169 lines)
├── position_encoding.py      # Positional encoding layers (239 lines)
├── backbones/               # Image feature extraction
│   ├── hieradet.py          # Hierarchical image encoder (317 lines)
│   ├── image_encoder.py     # Wrapper around backbone (134 lines)
│   └── utils.py             # Helper functions (93 lines)
└── sam/                     # From original SAM model
    ├── mask_decoder.py      # Mask decoding (295 lines)
    ├── transformer.py       # Transformer components (311 lines)
    └── prompt_encoder.py    # Prompt embedding (202 lines)
```

**Total Modeling Code: ~3,188 lines**

#### **`sam2/modeling/sam2_base.py`** (909 lines)
**Purpose:** Core model architecture combining all components

**Class: `SAM2Base(torch.nn.Module)`**

**Key Architecture Components:**
1. **Image Encoder** (Hierarchical backbone - HieraDet)
   - Multi-scale feature extraction
   - Adaptable depths/widths for model sizes
   
2. **Memory Encoder**
   - Processes previous frame masks into memory features
   - Temporal feature aggregation
   - Handles mask embedding and refinement

3. **Memory Attention**
   - Cross-temporal attention between frames
   - Efficient streaming memory updates
   - Supports conditioning frame selection

4. **Prompt Encoder** (from SAM)
   - Encodes point and box prompts
   - Dense positional encoding

5. **Mask Decoder** (from SAM)
   - Two-way transformer
   - Generates mask logits
   - Produces IoU predictions

**Key Initialization Parameters:**
- `num_maskmem` - Number of memory frames (default 7)
- `image_size` - Input resolution (default 512)
- `backbone_stride` - Feature map stride (default 16)
- `use_obj_ptrs_in_encoder` - Object pointer cross-attention
- `max_obj_ptrs_in_encoder` - Maximum tracked objects (default 16)
- `pred_obj_scores` - Predict object presence
- `memory_temporal_stride_for_eval` - Temporal downsampling
- `binarize_mask_from_pts_for_mem_enc` - Binary mask handling

**Forward Pass Architecture:**
1. Encode image to features
2. Process prompts through prompt encoder
3. Apply mask decoder with memory context
4. Generate mask logits and IoU scores
5. Post-process with stability/filtering

---

#### **`sam2/modeling/memory_encoder.py`** (181 lines)
- Encodes previous frame masks into memory features
- Handles mask-to-feature projection
- Supports mask binarization and scaling

#### **`sam2/modeling/memory_attention.py`** (169 lines)
- Cross-attention between current frame and memory frames
- Temporal attention with frame selection
- Efficient attention computation

#### **`sam2/modeling/position_encoding.py`** (239 lines)
- Positional encoding strategies for 2D spatial positions
- Temporal positional encoding for object pointers
- Supports absolute and relative positional encodings

#### **`sam2/modeling/backbones/hieradet.py`** (317 lines)
- Hierarchical image encoder (HierarchicalNet)
- Multi-scale feature pyramid
- Configurable depths for different model sizes

---

### 2.5 Utilities: `sam2/utils/`

#### **`sam2/utils/transforms.py`**
- Image preprocessing and normalization
- Resizing with aspect ratio preservation
- Mask post-processing (morphological operations)

#### **`sam2/utils/amg.py`**
- Automatic Mask Generator utilities
- Grid generation for point prompts
- Mask filtering and quality metrics

#### **`sam2/utils/misc.py`**
- Miscellaneous helper functions
- Video frame loading
- Mask processing utilities

---

## 3. Configuration System: `sam2/configs/`

Organized by model version and training stage:

```
sam2/configs/
├── sam2/                    # Original SAM 2 checkpoints (July 2024)
│   ├── sam2_hiera_t.yaml   # Tiny model (38.9M params)
│   ├── sam2_hiera_s.yaml   # Small model (46M params)
│   ├── sam2_hiera_b+.yaml  # Base+ model (80.8M params)
│   └── sam2_hiera_l.yaml   # Large model (224.4M params)
│
├── sam2.1/                  # Improved SAM 2.1 checkpoints (September 2024)
│   ├── sam2.1_hiera_t.yaml # Tiny (improved)
│   ├── sam2.1_hiera_s.yaml # Small (improved)
│   ├── sam2.1_hiera_b+.yaml # Base+ (improved)
│   └── sam2.1_hiera_l.yaml # Large (improved)
│
└── sam2.1_training/        # Fine-tuning configurations
    └── sam2.1_hiera_b+_MOSE_finetune.yaml
```

**YAML Configuration Format:**
- Hydra-based configuration composition
- Model architecture parameters
- Training hyperparameters
- Dataset paths and preprocessing
- Optimizer and scheduler settings

---

## 4. Training Infrastructure: `training/`

### **Purpose**
Complete training and fine-tuning pipeline for SAM 2 on custom datasets.

### **Directory Structure**

#### **`training/train.py`** (10,187 lines total, distributed)
- Launch script for training jobs
- Supports single-node and multi-node (SLURM cluster) training
- Configurable via command-line and YAML configs

#### **`training/trainer.py`** (41,374 lines)
- Main `Trainer` class orchestrating train/eval loop
- Distributed training support (torch.distributed)
- Checkpoint management and resumption
- Tensorboard logging
- Gradient accumulation and mixed precision

#### **`training/loss_fns.py`** (12,378 lines)
- `MultiStepMultiMasksAndIous` - Multi-step loss for iterative refinement
- Handles multiple mask outputs per frame
- IoU prediction losses
- Stability and matching losses

#### **`training/optimizer.py`** (19,873 lines)
- Flexible optimizer configuration with schedulers
- Layer-wise learning rate adjustment
- Warmup strategies
- Multiple optimizer backend support

#### **`training/dataset/`**
- **`sam2_datasets.py`** - Main dataset loader classes
- **`vos_dataset.py`** - Video Object Segmentation dataset wrapper
- **`vos_raw_dataset.py`** - Raw dataset loaders (SA-1B, SA-V, DAVIS-style)
- **`vos_sampler.py`** - Temporal sampling strategies
- **`vos_segment_loader.py`** - Mask/segment loading
- **`transforms.py`** - Data augmentation pipeline
- **`utils.py`** - Dataset utilities

Supports:
- SA-1B (image dataset)
- SA-V (video dataset with 51K videos)
- MOSE and DAVIS (video segmentation datasets)
- Custom datasets with similar structure

#### **`training/model/`**
- **`sam2.py`** - `SAM2Train` class extending `SAM2Base`
- Adds training-specific features:
  - Iterative point sampling simulation
  - Multi-mask output handling
  - Training-time augmentations

#### **`training/utils/`**
- **`checkpoint_utils.py`** - Checkpoint save/load
- **`distributed.py`** - Distributed training utilities
- **`logger.py`** - Tensorboard and file logging
- **`data_utils.py`** - Data loading utilities
- **`train_utils.py`** - Training helpers

---

## 5. Demonstration Applications

### 5.1 Full Web Demo: `demo/`

**Purpose:** Interactive web interface for image and video segmentation

**Architecture:**
- **Backend** (`demo/backend/`):
  - Flask application with GraphQL API via Strawberry
  - Processes video uploads and segmentation requests
  - Real-time video frame serving
  - Supports configurable model sizes
  
- **Frontend** (`demo/frontend/`):
  - React + TypeScript + Vite
  - Interactive canvas for drawing prompts
  - Real-time mask visualization
  - Frame navigation and timeline scrubbing

**Components:**
- `demo/backend/server/app.py` - Main Flask/GraphQL application
- `demo/backend/server/app_conf.py` - Configuration
- GraphQL schema for queries/mutations
- WebSocket support for real-time updates

**Deployment:**
- Docker containerization
- Docker Compose orchestration
- Supports CUDA, CPU, and MPS (Metal)

---

### 5.2 Lightweight Image Demo: `image_demo/`

**Purpose:** Simple, JSON-first single-image segmentation interface

**Architecture:**
- **Backend** (`image_demo/backend/app.py`):
  - Flask JSON API
  - Automatic mask generation
  - Lazy model loading per size
  - Caching for efficiency
  
- **Frontend** (`image_demo/frontend/`):
  - React + Vite
  - Image upload
  - Mask visualization with colors
  - JSON download for downstream pipelines

**API Endpoints:**
- `GET /health` - Service status
- `POST /segment-image` - Image segmentation with optional persistence

**Features:**
- Multi-model size support (tiny/small/base_plus/large)
- Configurable mask generation parameters
- Local JSON persistence option
- CORS-enabled for cross-origin requests

---

## 6. Dataset Utilities and Benchmarking

### 6.1 Segment Anything Video (SA-V) Dataset: `sav_dataset/`

**Purpose:** SA-V dataset management and evaluation

**Components:**
- **`sav_evaluator.py`** - Evaluation metrics for VOS tasks
- **`utils/sav_benchmark.py`** - Benchmark utilities
- **`utils/sav_utils.py`** - Dataset loading and processing

**Dataset Stats:**
- 51K diverse videos
- 643K spatio-temporal segmentation masks
- Training: 50,583 videos, 642,036 masklets
- Validation: 155 videos, 293 masklets
- Test: 150 videos, 278 masklets

**Supported Formats:**
- MP4 videos (24 fps)
- JPEG frames (extracted)
- JSON annotations (train)
- PNG masks (val/test, 6 fps)

---

### 6.2 Tools and Utilities: `tools/`

#### **`tools/vos_inference.py`** (22,397 lines)
- Inference script for Video Object Segmentation
- Generates predictions on VOS datasets
- Evaluates against benchmarks (DAVIS, MOSE, LVOS)
- Outputs in standard formats for evaluation

---

## 7. CUDA Extensions: `sam2/csrc/`

#### **`sam2/csrc/connected_components.cu`**
- GPU-accelerated connected components labeling
- Used for post-processing mask predictions
- Optional (Python fallback available)
- Improves performance on high-resolution outputs

---

## 8. Configuration Files

### **`setup.py`**
- Package metadata and dependencies
- Installation configuration
- Optional dependencies:
  - `notebooks` - Jupyter, matplotlib, OpenCV, eva-decord
  - `interactive-demo` - Flask, Strawberry GraphQL, av, gunicorn
  - `dev` - Black, testing, training dependencies
- CUDA extension compilation with error handling

### **`pyproject.toml`**
- Modern Python packaging configuration
- Build system requirements
- Entry points (if any)

### **Root Configuration Files**
- `docker-compose.yaml` - Multi-container orchestration
- `backend.Dockerfile` - Backend service container
- `.clang-format` - C++ code formatting rules
- `.gitignore` - VCS exclusions

---

## 9. Example Notebooks

Located in `notebooks/` with sample data in `notebooks/images/` and `notebooks/videos/`:

### **`image_predictor_example.ipynb`**
- Static image segmentation
- Interactive point/box prompting
- Mask refinement workflows
- Also available on Google Colab

### **`video_predictor_example.ipynb`**
- Video sequence processing
- Multi-object tracking
- Adding prompts at specific frames
- Propagating masks through time
- Also available on Google Colab

### **`automatic_mask_generator_example.ipynb`**
- Unsupervised dense mask generation
- Quality filtering parameters
- Batch processing
- Also available on Google Colab

---

## 10. Dependencies and Requirements

### **Core Requirements** (from `setup.py`)
```
torch>=2.5.1
torchvision>=0.20.1
numpy>=1.24.4
tqdm>=4.66.1
hydra-core>=1.3.2
iopath>=0.1.10
pillow>=9.4.0
```

### **Optional: Notebooks**
```
matplotlib>=3.9.1
jupyter>=1.0.0
opencv-python>=4.7.0
eva-decord>=0.6.1
```

### **Optional: Interactive Demo**
```
Flask>=3.0.3
Flask-Cors>=5.0.0
av>=13.0.0
dataclasses-json>=0.6.7
gunicorn>=23.0.0
imagesize>=1.4.1
pycocotools>=2.0.8
strawberry-graphql>=0.243.0
```

### **Optional: Development & Training**
```
black==24.2.0
usort==1.0.2
ufmt==2.0.0b2
fvcore>=0.1.5
pandas>=2.2.2
scikit-image>=0.24.0
tensorboard>=2.17.0
tensordict>=0.6.0
submitit>=1.5.1
```

---

## 11. Architecture Patterns and Design Decisions

### **Hydra Configuration System**
- Configuration composition and overrides
- Factory pattern via instantiate
- Enables modular model/dataset/optimizer swapping

### **Inference State Management**
- Stateful video processing with `inference_state` dictionary
- Enables streaming and interactive refinement
- Memory-efficient frame offloading options

### **Modular Model Components**
- Clear separation: encoder, decoder, memory, attention
- Pluggable components (different backbones, heads)
- Training-specific variants via inheritance

### **Multi-Scale Processing**
- Hierarchical backbone for efficiency
- Crop-based processing for high-resolution inputs
- Adaptive memory compression for temporal efficiency

### **Flexible API Design**
- Class methods for Hugging Face integration
- Support for both eager and state-based interfaces
- Backward compatibility with legacy implementations

---

## 12. Key Features and Capabilities

### **Image Segmentation**
- Promptable segmentation (points, boxes)
- Automatic dense mask generation
- Interactive refinement

### **Video Segmentation**
- Multi-object tracking
- Streaming memory for real-time processing
- Temporal consistency
- Independent per-object inference

### **Model Variants**
- 4 sizes: tiny (38.9M), small (46M), base+ (80.8M), large (224.4M)
- 2 versions: SAM 2 original and improved SAM 2.1
- Hierarchical architecture with configurable depths

### **Training Support**
- Fine-tuning on custom datasets
- Mixed image and video training
- Multi-node distributed training (SLURM)
- Comprehensive loss functions and optimization

### **Deployment**
- Docker containerization
- Multiple device support (CUDA, CPU, MPS)
- GraphQL and JSON APIs
- Lazy model loading with caching

---

## 13. Testing Infrastructure

**Status:** No dedicated test suite in main directories
- Testing likely handled via notebooks and inference scripts
- Quality assurance through benchmarking tools (`benchmark.py`)
- Validation scripts in `sav_dataset/` for dataset evaluation

---

## 14. Documentation Structure

### **Primary References**
- `README.md` - Project overview and quick start
- `INSTALL.md` - Detailed installation instructions
- `RELEASE_NOTES.md` - Version history and changes
- `training/README.md` - Training code documentation
- `demo/README.md` - Web demo setup
- `image_demo/README.md` - Image demo setup
- `tools/README.md` - Tool documentation
- `sav_dataset/README.md` - Dataset documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `CODE_OF_CONDUCT.md` - Community standards

---

## 15. Recent Updates (as of Latest Release)

### **December 11, 2024**
- Torch.compile support for entire SAM2 model (video mode)
- New `SAM2VideoPredictorVOS` class for optimized VOS
- Independent per-object inference capability

### **September 30, 2024 (SAM 2.1 Release)**
- Improved checkpoints with better performance
- Official training code release
- Web demo frontend/backend code
- Updated model configs and documentation

---

## 16. Quick Navigation Guide

**Want to...**
- **Segment static images?** → `sam2/sam2_image_predictor.py`
- **Track objects in video?** → `sam2/sam2_video_predictor.py`
- **Generate dense masks?** → `sam2/automatic_mask_generator.py`
- **Fine-tune models?** → `training/train.py`, `training/trainer.py`
- **Deploy web interface?** → `demo/` or `image_demo/`
- **Evaluate on datasets?** → `sav_dataset/sav_evaluator.py`, `tools/vos_inference.py`
- **Load pre-trained models?** → `sam2/build_sam.py`
- **Understand architecture?** → `sam2/modeling/sam2_base.py`

---

## 17. Codebase Statistics

| Component | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| Core Modeling | 10 | 3,188 | Neural network architecture |
| Main APIs | 3 | 2,161 | Image/video prediction |
| Utilities | 5+ | 1,000+ | Helper functions |
| Training | 20+ | 85,000+ | Training pipeline |
| Demo (web) | 10+ | 5,000+ | Full web interface |
| Demo (image) | 5+ | 2,000+ | Lightweight JSON API |
| Tools & Eval | 5+ | 30,000+ | Benchmarking and evaluation |
| Configs | 10+ | YAML | Model and training configs |

**Total Python Code (excluding venv):** ~130,000+ lines

---

## Summary for Future Instances

This codebase represents a production-ready, research-focused foundation model framework with:

1. **Clear Architecture:** Well-separated concerns between modeling, inference, training, and deployment
2. **Multiple Entry Points:** Image inference, video tracking, automatic generation, training, and web interfaces
3. **Production Ready:** Docker support, comprehensive APIs, error handling, and documentation
4. **Research-Friendly:** Modular components, Hydra configuration, extensible for custom modifications
5. **Scalable:** Distributed training, multi-object tracking, efficient memory management

Key strengths for Claude Code navigation:
- Main model logic concentrated in `sam2/modeling/`
- Clear separation between legacy and current implementations
- Extensive examples in Jupyter notebooks
- Well-documented configuration system
- Straightforward factory functions in `build_sam.py`


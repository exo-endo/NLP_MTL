# SDOH MTL Project Setup Guide

This guide will help you set up the SDOH Multi-Task Learning project on a new computer.

## Prerequisites

- Python 3.13 (or 3.10+)
- At least 8GB RAM
- For GPU acceleration: CUDA-capable GPU or Apple Silicon (M1/M2/M3)

## Quick Setup

### 1. Create Virtual Environment

```bash
cd hlessclass
python3 -m venv hless
source hless/bin/activate  # On Windows: hless\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you encounter issues with torch/MPS on non-Apple Silicon:
```bash
# For CUDA (NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Project Structure

```
hlessclass/
├── Data/                           # Dataset files
│   ├── MTL_preprocessed.csv       # Main training data
│   └── ...
├── models/                         # Trained models
│   ├── clinicalbert_sdoh_mtl/     # MTL model
│   └── clinicalbert_homeless/     # Binary classifier
├── apps/                           # Streamlit applications
│   ├── streamlit_explain_mtl.py   # MTL explainer app
│   └── streamlit_explain.py       # Binary explainer app
├── wordclouds/                     # Generated word clouds
├── train_clinicalbert.py          # Training script
├── generate_wordclouds.py         # Word cloud generator
├── run_streamlit_mtl.py           # MTL app launcher
├── run_streamlit.py               # Binary app launcher
├── requirements.txt               # Python dependencies
└── SETUP.md                       # This file
```

## Usage

### Training the Model

```bash
# Activate environment
source hless/bin/activate

# Train with default settings (2 epochs)
python train_clinicalbert.py --epochs 2

# Train with custom settings
python train_clinicalbert.py \
  --data Data/MTL_preprocessed.csv \
  --out models/clinicalbert_sdoh_mtl \
  --model emilyalsentzer/Bio_ClinicalBERT \
  --epochs 5
```

### Running Streamlit Apps

**MTL Explainer (3 SDOH domains):**
```bash
python run_streamlit_mtl.py
# Opens at http://localhost:8502
```

**Binary Classifier (Homeless only):**
```bash
python run_streamlit.py
# Opens at http://localhost:8501
```

### Generating Word Clouds

```bash
# Test with 10 samples
python generate_wordclouds.py --max-samples 10

# Full run (takes ~11 hours for 3000 samples)
python generate_wordclouds.py --max-samples 3000

# Process all data
python generate_wordclouds.py
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### Memory Issues

For systems with limited RAM:
- Reduce batch size in training: `--per_device_train_batch_size=4`
- Use CPU instead of GPU: Set `CUDA_VISIBLE_DEVICES=""`
- Process fewer samples for word clouds

### Model Not Found

Ensure the model directory exists:
```bash
ls -la models/clinicalbert_sdoh_mtl/
```

If missing, you need to train the model first or copy from the original machine.

## Moving to a New Computer

### What to Copy

**Essential files (small):**
- All `.py` scripts
- `requirements.txt`
- `SETUP.md`
- `Data/` folder (if manageable size)
- `demo.svg` (logo)

**Large files (optional):**
- `models/` folder (~400MB per model) - can retrain if needed
- `wordclouds/` folder - can regenerate

### Steps

1. Copy the project folder to new computer
2. Follow "Quick Setup" steps above
3. If models are missing, retrain or copy from original machine
4. Run verification tests

## System Requirements

**Minimum:**
- 8GB RAM
- 20GB free disk space
- Python 3.10+

**Recommended:**
- 16GB+ RAM
- GPU with 6GB+ VRAM (or Apple Silicon)
- 50GB free disk space
- Python 3.13

## Support

For issues or questions, check:
- Log files: `log.txt`, `wordcloud_run.log`, `train_run.log`
- Model configs: `models/*/config.json`
- Data format: First few rows of `Data/MTL_preprocessed.csv`

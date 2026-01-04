**Project:** hlessclass — ClinicalBERT fine-tuning + Streamlit explain app

- **Purpose:** Fine-tune ClinicalBERT on the preprocessed dataset and provide an interactive Streamlit app to visualize attributions.

**Quick paths**
- Project root: `hlessclass`
- Preprocessed data: `Data/preprocessed.csv`
- Trained model dir: `models/clinicalbert_homeless` (contains `model.safetensors`, `tokenizer.json`, etc.)
- Training script: `train_clinicalbert.py`
- Streamlit app: `apps/streamlit_explain.py`
- Streamlit launcher (no shell script): `run_streamlit.py`
- Log file: `log.txt` (contains timestamped training summary lines)

**Prerequisites**
- macOS, zsh
- Use the bundled venv at `hless/` or install required packages into your own environment. The repo includes a full `hless` venv for reproducibility.

**Recommended: run Streamlit only (no retrain)**
1. Open one zsh terminal and activate the venv:

```bash
cd "/Users/s3cr3tmyth/Hobby Projects/hlessclass"
source hless/bin/activate
```

2. Launch Streamlit using the bundled Python (recommended to avoid shebang issues):

```bash
"/Users/s3cr3tmyth/Hobby Projects/hlessclass/hless/bin/python" run_streamlit.py
```

3. Open the app in your browser: `http://localhost:8501`.

**Run training (only if you want to re-fine-tune)**
- Train for 2 epochs (example):

```bash
cd "/Users/s3cr3tmyth/Hobby Projects/hlessclass"
source hless/bin/activate
"/Users/s3cr3tmyth/Hobby Projects/hlessclass/hless/bin/python" train_clinicalbert.py --epochs 2
```

- After training finishes the model is saved to `models/clinicalbert_homeless` and a concise entry is appended to `log.txt`.

**Run both sequentially in one terminal (train then serve)**
```bash
cd "/Users/s3cr3tmyth/Hobby Projects/hlessclass"
source hless/bin/activate
"/Users/s3cr3tmyth/Hobby Projects/hlessclass/hless/bin/python" train_clinicalbert.py --epochs 2 && \
"/Users/s3cr3tmyth/Hobby Projects/hlessclass/hless/bin/python" run_streamlit.py
```

**Run Streamlit detached (keeps running after terminal close)**
- Using `nohup`:

```bash
cd "/Users/s3cr3tmyth/Hobby Projects/hlessclass"
source hless/bin/activate
nohup "/Users/s3cr3tmyth/Hobby Projects/hlessclass/hless/bin/python" run_streamlit.py > streamlit.log 2>&1 &
```

- Or use `tmux` / `screen` for a managed session.

**Troubleshooting**
- If Streamlit tries to download a model from Hugging Face, confirm the app is loading the local model dir: `models/clinicalbert_homeless` and that files like `model.safetensors`, `tokenizer.json`, `config.json` exist.
- If `python` isn't found after `source hless/bin/activate`, call the venv Python explicitly as shown above.

If you want, I can add a one-line helper script `run_all.sh` (train+serve) or generate `requirements.txt` so the `hless/` venv isn't required. Let me know which you prefer.

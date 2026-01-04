import os

# --------------------------------------------------
# Force offline mode BEFORE importing transformers
# --------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Optional: point HF cache somewhere writable
# os.environ["HF_HOME"] = "/path/to/hf_cache"

import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from captum.attr import IntegratedGradients

# Paths (resolve to absolute paths so transformers treats them as local dirs)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CSV = os.path.join(BASE_DIR, 'Data', 'MTL_preprocessed.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'clinicalbert_sdoh_mtl')

# --------------------------------------------------
# CHANGE: Use a LOCAL base model directory for tokenizer + encoder
# Put the downloaded HF model folder here (must contain config.json, vocab/tokenizer files, weights)
# Example:
#   BASE_MODEL_DIR = os.path.join(BASE_DIR, 'hf_models', 'Bio_ClinicalBERT')
# --------------------------------------------------
BASE_MODEL_DIR = os.path.join(BASE_DIR, 'hf_models', 'Bio_ClinicalBERT')

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Custom MTL model class (must match training script)
class ClinicalBERT_MTL(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, local_only: bool = True):
        super().__init__()

        # CHANGE: load config + encoder locally (no internet)
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
        self.bert = AutoModel.from_pretrained(
            model_name,
            config=config,
            local_files_only=local_only
        )

        hidden = self.bert.config.hidden_size

        # One classification head per SDOH domain
        self.environment = torch.nn.Linear(hidden, num_labels)
        self.education   = torch.nn.Linear(hidden, num_labels)
        self.economics   = torch.nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
    ):
        if inputs_embeds is not None:
            outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        cls_emb = outputs.last_hidden_state[:, 0]  # [CLS] token

        logits_env  = self.environment(cls_emb)
        logits_edu  = self.education(cls_emb)
        logits_econ = self.economics(cls_emb)

        return {
            "logits_env": logits_env,
            "logits_edu": logits_edu,
            "logits_econ": logits_econ,
        }

@st.cache_resource
def load_model_and_tokenizer():
    # CHANGE: local tokenizer load
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)

    # CHANGE: local encoder init (heads loaded from your trained checkpoint below)
    model = ClinicalBERT_MTL(model_name=BASE_MODEL_DIR, num_labels=3, local_only=True)

    # Load saved weights (unchanged)
    state_dict_path = os.path.join(MODEL_DIR, 'model.safetensors')
    if os.path.exists(state_dict_path):
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path)
        model.load_state_dict(state_dict)
    else:
        # Try pytorch_model.bin as fallback
        pt_path = os.path.join(MODEL_DIR, 'pytorch_model.bin')
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            st.error(f"Could not find model weights at {MODEL_DIR}")
            st.stop()

    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_data
def load_data():
    if not os.path.exists(DATA_CSV):
        return None
    df = pd.read_csv(DATA_CSV)
    return df

def predict_and_attr(
    text,
    tokenizer,
    model,
    sdoh_domain='environment',
    target_label=None,
    max_length=256,
    chunk_threshold=384,   # ✅ only chunk if longer than this
):
    domain_map = {
        'environment': 'logits_env',
        'education': 'logits_edu',
        'economics': 'logits_econ'
    }
    logits_key = domain_map[sdoh_domain]

    num_tokens = count_tokens(text, tokenizer)

    # --------------------------------------------------
    # SINGLE PASS (unchanged behavior)
    # --------------------------------------------------
    def run_single_pass(txt):
        enc = tokenizer(
            txt,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[logits_key]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return enc, logits, probs

    # --------------------------------------------------
    # Decide path
    # --------------------------------------------------
    if num_tokens <= chunk_threshold:
        enc, logits, probs = run_single_pass(text)
        best_enc = enc

    else:
        chunks = chunk_text(text, tokenizer)
        chunk_results = []

        for ch in chunks:
            enc, logits, probs = run_single_pass(ch)
            chunk_results.append({
                "enc": enc,
                "probs": probs
            })

        # ✅ Aggregate using max prob for adverse class (2)
        best_idx = int(np.argmax([r["probs"][2] for r in chunk_results]))
        best_enc = chunk_results[best_idx]["enc"]
        probs = chunk_results[best_idx]["probs"]

    # --------------------------------------------------
    # Captum attribution on best chunk
    # --------------------------------------------------
    pred_label = int(np.argmax(probs))
    if target_label is None:
        target_label = pred_label

    input_ids = best_enc["input_ids"].to(device)
    attention_mask = best_enc["attention_mask"].to(device)

    embeddings = model.bert.get_input_embeddings()(input_ids)

    def forward_func(inputs_embeds, attention_mask):
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out[logits_key][:, target_label]

    ig = IntegratedGradients(forward_func)
    attributions, delta = ig.attribute(
        embeddings,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True
    )

    token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    return tokens, token_attr, probs, delta, num_tokens


def colorize(tokens, attributions, max_display=256):
    # Simple symmetric normalization and red↔white↔green mapping
    vals = np.array(attributions)
    # Ignore special tokens when computing scale
    mask = np.array([t not in ['[CLS]', '[SEP]', '[PAD]'] for t in tokens])
    if mask.sum() == 0:
        scale = 1.0
    else:
        m = np.max(np.abs(vals[mask]))
        scale = m if m != 0 else 1.0

    def score_to_rgba(s):
        # s is in [-1,1]
        s = max(-1.0, min(1.0, s))
        alpha = 0.65
        if s >= 0:
            # interpolate white -> red
            r = 255
            g = int(255 * (1.0 - s))
            b = int(255 * (1.0 - s))
        else:
            # interpolate white -> green
            n = -s
            g = 255
            r = int(255 * (1.0 - n))
            b = int(255 * (1.0 - n))
        return f'rgba({r},{g},{b},{alpha})'

    spans = []
    for t, v in zip(tokens, vals):
        if t in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        s = float(v) / scale
        color = score_to_rgba(s)
        # Cleanup token display
        tok = t
        if tok.startswith('Ġ'):
            tok = tok[1:]
        tok = tok.replace('▁', '')
        tok = tok.replace('##', '')
        spans.append(f"<span style='background:{color}; padding:2px; margin:1px;border-radius:4px'>{tok}</span>")

    # Build a CSS gradient legend (green -> white -> red) and numeric min/0/max
    minv = float(-scale)
    maxv = float(scale)
    legend_html = f"""
    <div style='width:100%; padding:6px;'>
        <div style='height:18px; background: linear-gradient(to right, rgba(0,200,0,0.65), rgba(255,255,255,0.8), rgba(255,0,0,0.65)); border-radius:4px;'></div>
        <div style='display:flex; justify-content:space-between; font-size:12px; color:#fff; margin-top:6px; font-weight:600; text-shadow: 0 1px 0 rgba(0,0,0,0.4);'>
            <div>{minv:.3f}</div>
            <div>0.000</div>
            <div>{maxv:.3f}</div>
        </div>
    </div>
    """

    return ' '.join(spans), legend_html

# ----------------------------
# Chunking helpers
# ----------------------------
def count_tokens(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def chunk_text(text, tokenizer, chunk_size=384, stride=128):
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    chunks = []
    for start in range(0, len(token_ids), chunk_size - stride):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)
        if end >= len(token_ids):
            break

    return chunks


# Streamlit UI
st.set_page_config(page_title='ClinicalBERT MTL SDOH Explainer', layout='wide')

# Header with logo
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title('SDOH - Predictions and Explainer tool v.0.01')
with header_col2:
    logo_path = os.path.join(BASE_DIR, 'demo.svg')
    if os.path.exists(logo_path):
        with open(logo_path, 'r') as f:
            svg_content = f.read()
        # Wrap SVG with constrained width and height
        st.markdown(f'''
        <div style="text-align: right; padding-top: 10px;">
            <div style="width: 120px; height: 60px; margin-left: auto; overflow: hidden;">
                <div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                    {svg_content}
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

tokenizer, model = load_model_and_tokenizer()
df = load_data()

# SDOH Domain Selection
st.sidebar.header('SDOH Domain Selection')
sdoh_domain = st.sidebar.radio(
    'Select SDOH Domain to Explain',
    options=['environment', 'education', 'economics'],
    index=0,
    help='Choose which Social Determinants of Health domain to analyze'
)

# Display domain info
domain_info = {
    'environment': {
        'name': 'Environment (Housing/Homelessness)',
        'labels': ['No mention', 'Stable housing', 'Housing insecurity/Homeless'],
        'color': '#4CAF50'
    },
    'education': {
        'name': 'Education',
        'labels': ['No mention', 'Some education noted', 'Educational challenges'],
        'color': '#2196F3'
    },
    'economics': {
        'name': 'Economics (Financial Status)',
        'labels': ['No mention', 'Financial stability', 'Financial insecurity'],
        'color': '#FF9800'
    }
}

st.sidebar.markdown(f"**Selected:** {domain_info[sdoh_domain]['name']}")
st.sidebar.markdown(f"**Label meanings:**")
for i, label in enumerate(domain_info[sdoh_domain]['labels']):
    st.sidebar.markdown(f"- **{i}:** {label}")

# Input source: dataset row or custom text
st.subheader('Input')
input_source = st.radio('Input source', ('Dataset row', 'Custom text'), index=0, horizontal=True)

if input_source == 'Dataset row':
    if df is not None:
        idx = st.number_input('Pick test row index (0..n-1)', min_value=0, max_value=len(df)-1, value=0)
        row = df.iloc[idx]
        text = row.get('text', row.get('social_history', ''))
        if not text:
            text = str(row.iloc[0]) if len(row) > 0 else ''

        # Get true labels for all domains
        true_env = row.get('environment_label', row.get('sdoh_environment', None))
        true_edu = row.get('education_label', row.get('sdoh_education', None))
        true_econ = row.get('economics_label', row.get('sdoh_economics', None))
    else:
        st.warning('Dataset not found. Using custom text input.')
        text = ''
        true_env = true_edu = true_econ = None
else:
    text = st.text_area('Enter custom clinical text to explain', value='', height=220)
    true_env = true_edu = true_econ = None

st.subheader('Clinical Text')
st.write(text if text else '_(empty)_')

col1, col2 = st.columns([3, 1])

# Create a placeholder in the legend column
with col2:
    st.markdown('**Attribution Legend**')
    legend_placeholder = st.empty()
    legend_placeholder.markdown("<div style='color:#666'>Click 'Explain this note' to show attribution colors.</div>", unsafe_allow_html=True)

with col1:
    if st.button('Explain this note', type='primary'):
        if not text:
            st.warning('No text provided to explain.')
        else:
            with st.spinner(f'Computing attributions for {sdoh_domain}...'):
                tokens, token_attr, probs, delta, num_tokens = predict_and_attr(text, tokenizer, model, sdoh_domain=sdoh_domain)
                pred_label = int(np.argmax(probs))
                st.markdown(f"_Input length: {num_tokens} tokens — "f"{'chunked' if num_tokens > 384 else 'single-pass'} inference_")

                st.markdown(f"### Results for **{domain_info[sdoh_domain]['name']}**")
                st.markdown(f"**Predicted Label:** {pred_label} — _{domain_info[sdoh_domain]['labels'][pred_label]}_")
                st.markdown(f"**Model Probabilities:**")
                for i, p in enumerate(probs):
                    st.markdown(f"- Label {i}: {p:.3f}")

                st.markdown("### Token Attributions")
                html, legend_html = colorize(tokens, token_attr)
                st.markdown(html, unsafe_allow_html=True)

                # Render the shaded legend in the right column
                if legend_html:
                    legend_placeholder.markdown(legend_html, unsafe_allow_html=True)

                st.markdown(f"**Convergence Delta:** {float(delta):.6f}")

                # Show true labels if available
                if input_source == 'Dataset row' and df is not None:
                    st.markdown("### True Labels (from dataset)")
                    true_labels = {
                        'environment': true_env,
                        'education': true_edu,
                        'economics': true_econ
                    }
                    for domain, true_val in true_labels.items():
                        if true_val is not None:
                            label_str = domain_info[domain]['labels'][int(true_val)]
                            highlight = " ✓" if domain == sdoh_domain else ""
                            st.markdown(f"- **{domain_info[domain]['name']}:** {int(true_val)} — _{label_str}_{highlight}")

st.markdown('---')
st.markdown("""
**How to use this tool:**
1. Select a SDOH domain from the sidebar (Environment, Education, or Economics)
2. Choose input source: load from dataset or enter custom text
3. Click "Explain this note" to see model predictions and token-level attributions
4. Red highlights indicate tokens that support the predicted label
5. Green highlights indicate tokens that oppose the predicted label

**Note:** This uses IntegratedGradients over input embeddings. Attributions are computed for the selected SDOH domain's predicted label.
""")

import os
# --------------------------------------------------
# Force offline mode BEFORE importing transformers
# --------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Optional: point HF cache somewhere writable
# os.environ["HF_HOME"] = "/path/to/hf_cache"

import argparse
import datetime
import pandas as pd
import torch
import torch.nn as nn
import numpy as np  # NEW

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix  # NEW
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
)

# --------------------------------------------------
# Multi-task ClinicalBERT model
# --------------------------------------------------
class ClinicalBERT_MTL(nn.Module):
    def __init__(
        self,
        model_source: str,
        num_labels: int = 3,
        local_only: bool = True,
        env_weights=None,   # NEW
        edu_weights=None,   # NEW
        econ_weights=None,  # NEW
    ):
        """
        model_source: local directory path containing config.json, tokenizer files, model weights, etc.
                    e.g. "/models/Bio_ClinicalBERT"
                    (Can also be a HF repo id, but local_only=True will prevent online fetch.)
        """
        super().__init__()

        # Load config + model locally (no internet)
        config = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
        self.bert = AutoModel.from_pretrained(
            model_source,
            config=config,
            local_files_only=local_only
        )

        hidden = self.bert.config.hidden_size

        # One classification head per SDOH domain
        self.environment = nn.Linear(hidden, num_labels)
        self.education   = nn.Linear(hidden, num_labels)
        self.economics   = nn.Linear(hidden, num_labels)

        # NEW: store weights as buffers so they move with model.to(device)
        self.register_buffer("env_weights", env_weights if env_weights is not None else None, persistent=False)
        self.register_buffer("edu_weights", edu_weights if edu_weights is not None else None, persistent=False)
        self.register_buffer("econ_weights", econ_weights if econ_weights is not None else None, persistent=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        environment_label=None,
        education_label=None,
        economics_label=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_emb = outputs.last_hidden_state[:, 0]  # [CLS] token

        logits_env  = self.environment(cls_emb)
        logits_edu  = self.education(cls_emb)
        logits_econ = self.economics(cls_emb)

        loss = None
        if environment_label is not None:
            # NEW: class-weighted CE per head (falls back to unweighted if weights are None)
            loss_env = nn.CrossEntropyLoss(weight=self.env_weights)(logits_env, environment_label) \
                if self.env_weights is not None else nn.CrossEntropyLoss()(logits_env, environment_label)

            loss_edu = nn.CrossEntropyLoss(weight=self.edu_weights)(logits_edu, education_label) \
                if self.edu_weights is not None else nn.CrossEntropyLoss()(logits_edu, education_label)

            loss_econ = nn.CrossEntropyLoss(weight=self.econ_weights)(logits_econ, economics_label) \
                if self.econ_weights is not None else nn.CrossEntropyLoss()(logits_econ, economics_label)

            loss = loss_env + loss_edu + loss_econ

        return {
            "loss": loss,
            "logits_env": logits_env,
            "logits_edu": logits_edu,
            "logits_econ": logits_econ,
        }


# NEW: class-weight helper
def compute_class_weights(series, num_classes=3):
    counts = series.value_counts().reindex(range(num_classes), fill_value=0).to_numpy(dtype=np.float32)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero
    weights = counts.sum() / counts   # inverse frequency
    weights = weights / weights.mean()  # normalize
    return torch.tensor(weights, dtype=torch.float32), counts.astype(int)


# NEW: pretty confusion matrix formatting
def format_cm(cm):
    lines = []
    lines.append("        pred0  pred1  pred2")
    for i, row in enumerate(cm):
        lines.append(f"true{i}  {row[0]:>6} {row[1]:>6} {row[2]:>6}")
    return "\n".join(lines)


# --------------------------------------------------
# Main training function
# --------------------------------------------------
def main(data_csv: str, out_dir: str, model_source: str, epochs: int):
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # Load CSV
    # ----------------------------
    df = pd.read_csv(data_csv)

    # Normalize text column
    if "text" in df.columns:
        df["text"] = df["text"].fillna("")
    elif "social_history" in df.columns:
        df["text"] = df["social_history"].fillna("")
    else:
        raise ValueError("Could not find a text column.")

    # ----------------------------
    # Normalize label column names
    # ----------------------------
    df = df.rename(columns={
        "sdoh_environment": "environment_label",
        "sdoh_education":   "education_label",
        "sdoh_economics":   "economics_label",
    })

    label_cols = [
        "environment_label",
        "education_label",
        "economics_label",
    ]

    # Validate label columns
    for col in label_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required label column: {col}")
        df[col] = df[col].astype(int)

        # Sanity check allowed values
        if not set(df[col].unique()).issubset({0, 1, 2}):
            raise ValueError(f"Invalid values found in {col}")

    # ----------------------------
    # Train / test split
    # ----------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["environment_label"],  # anchor on one task
    )

    # NEW: compute per-head weights from TRAIN split only
    w_env, env_counts   = compute_class_weights(train_df["environment_label"])
    w_edu, edu_counts   = compute_class_weights(train_df["education_label"])
    w_econ, econ_counts = compute_class_weights(train_df["economics_label"])

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # ----------------------------
    # Tokenizer (LOCAL)
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    test_dataset  = test_dataset.map(tokenize_fn, batched=True)

    # Keep only tensors needed for training
    columns_to_keep = [
        "input_ids",
        "attention_mask",
        "environment_label",
        "education_label",
        "economics_label",
    ]

    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in columns_to_keep]
    )
    test_dataset = test_dataset.remove_columns(
        [c for c in test_dataset.column_names if c not in columns_to_keep]
    )

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # ----------------------------
    # Model (LOCAL)
    # ----------------------------
    # NEW: pass weights into model
    model = ClinicalBERT_MTL(
        model_source=model_source,
        num_labels=3,
        local_only=True,
        env_weights=w_env,
        edu_weights=w_edu,
        econ_weights=w_econ,
    )

    # ----------------------------
    # Metrics (keep minimal: accuracy only)
    # ----------------------------
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        logits_env, logits_edu, logits_econ = preds
        y_env, y_edu, y_econ = labels

        # Convert to numpy if they're torch tensors
        if hasattr(logits_env, "cpu"):
            logits_env = logits_env.cpu().numpy()
            logits_edu = logits_edu.cpu().numpy()
            logits_econ = logits_econ.cpu().numpy()
            y_env = y_env.cpu().numpy()
            y_edu = y_edu.cpu().numpy()
            y_econ = y_econ.cpu().numpy()

        return {
            "env_acc": accuracy_score(y_env, logits_env.argmax(-1)),
            "edu_acc": accuracy_score(y_edu, logits_edu.argmax(-1)),
            "econ_acc": accuracy_score(y_econ, logits_econ.argmax(-1)),
        }

    # ----------------------------
    # Training arguments
    # ----------------------------
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        logging_steps=50,
        eval_strategy="epoch",  # fixed parameter name
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    # ----------------------------
    # Custom Trainer for MTL
    # ----------------------------
    class MTLTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                environment_label=inputs["environment_label"],
                education_label=inputs["education_label"],
                economics_label=inputs["economics_label"],
            )
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            logits = (
                outputs["logits_env"].cpu(),
                outputs["logits_edu"].cpu(),
                outputs["logits_econ"].cpu(),
            )

            labels = (
                inputs["environment_label"].cpu(),
                inputs["education_label"].cpu(),
                inputs["economics_label"].cpu(),
            )

            return (None, logits, labels)

    trainer = MTLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ----------------------------
    # Train & evaluate
    # ----------------------------
    trainer.train()
    res = trainer.evaluate()
    print(res)

    trainer.save_model(out_dir)
    print("Saved MTL ClinicalBERT to:", out_dir)

    # ----------------------------
    # NEW: write one timestamped eval log + confusion matrices at the end
    # ----------------------------
    pred_out = trainer.predict(test_dataset)
    logits_env, logits_edu, logits_econ = pred_out.predictions
    y_env, y_edu, y_econ = pred_out.label_ids

    p_env  = np.argmax(logits_env, axis=-1)
    p_edu  = np.argmax(logits_edu, axis=-1)
    p_econ = np.argmax(logits_econ, axis=-1)

    cm_env  = confusion_matrix(y_env,  p_env,  labels=[0, 1, 2]).tolist()
    cm_edu  = confusion_matrix(y_edu,  p_edu,  labels=[0, 1, 2]).tolist()
    cm_econ = confusion_matrix(y_econ, p_econ, labels=[0, 1, 2]).tolist()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"eval_{ts}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Data: {data_csv}\n")
        f.write(f"Model source: {model_source}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Epochs: {epochs}\n\n")

        f.write("Train label counts (0/1/2):\n")
        f.write(f"  env:  {env_counts.tolist()}\n")
        f.write(f"  edu:  {edu_counts.tolist()}\n")
        f.write(f"  econ: {econ_counts.tolist()}\n\n")

        f.write("Class weights:\n")
        f.write(f"  env:  {w_env.tolist()}\n")
        f.write(f"  edu:  {w_edu.tolist()}\n")
        f.write(f"  econ: {w_econ.tolist()}\n\n")

        f.write("Final eval metrics:\n")
        for k in sorted(res.keys()):
            f.write(f"  {k}: {res[k]}\n")

        f.write("\nConfusion matrices (rows=true, cols=pred):\n\n")
        f.write("[ENV]\n")
        f.write(format_cm(cm_env) + "\n\n")
        f.write("[EDU]\n")
        f.write(format_cm(cm_edu) + "\n\n")
        f.write("[ECON]\n")
        f.write(format_cm(cm_econ) + "\n")

    print("Wrote eval summary to:", log_path)


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Data/MTL_preprocessed.csv")
    parser.add_argument("--out", type=str, default="models/clinicalbert_sdoh_mtl")

    # IMPORTANT:
    # Pass a LOCAL DIRECTORY on your work laptop, e.g.:
    #   --model /models/Bio_ClinicalBERT
    #
    # Default below is a placeholder local path; change it to your real folder.
    parser.add_argument("--model", type=str, default="hf_models/Bio_ClinicalBERT")

    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    main(args.data, args.out, args.model, args.epochs)

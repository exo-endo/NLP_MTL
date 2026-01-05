import os
# --------------------------------------------------
# Force offline mode BEFORE importing transformers
# --------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Optional: point HF cache somewhere writable
# os.environ["HF_HOME"] = "/path/to/hf_cache"

# --------------------------------------------------
# Reduce deprecation/future warning noise from transformers
# --------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"transformers")

import argparse
import datetime
import math
import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,  # NEW
)
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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
        env_weights=None,
        edu_weights=None,
        econ_weights=None,
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(model_source, local_files_only=local_only)
        self.bert = AutoModel.from_pretrained(
            model_source,
            config=config,
            local_files_only=local_only
        )

        hidden = self.bert.config.hidden_size

        self.environment = nn.Linear(hidden, num_labels)
        self.education   = nn.Linear(hidden, num_labels)
        self.economics   = nn.Linear(hidden, num_labels)

        # store weights as buffers so they move with model.to(device)
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0]

        logits_env  = self.environment(cls_emb)
        logits_edu  = self.education(cls_emb)
        logits_econ = self.economics(cls_emb)

        loss = None
        if environment_label is not None:
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


# --------------------------------------------------
# Logging helpers
# --------------------------------------------------
def _fmt_cm(cm_3x3):
    # Pretty 3x3 confusion matrix lines
    # rows=true, cols=pred
    lines = []
    lines.append("        pred0  pred1  pred2")
    for i, row in enumerate(cm_3x3):
        lines.append(f"true{i}  {row[0]:>6} {row[1]:>6} {row[2]:>6}")
    return "\n".join(lines)

def _write_log(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# --------------------------------------------------
# Callback: print progress + write readable logs to log.txt
# --------------------------------------------------
class ProgressAndFileLoggerCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.last_eval = {}

    def on_train_begin(self, args, state, control, **kwargs):
        _write_log(self.log_path, "=" * 80)
        _write_log(self.log_path, f"RUN START: {datetime.datetime.now().isoformat()}")
        _write_log(self.log_path, f"output_dir: {args.output_dir}")
        _write_log(self.log_path, f"num_train_epochs: {args.num_train_epochs}")
        _write_log(self.log_path, f"per_device_train_batch_size: {args.per_device_train_batch_size}")
        _write_log(self.log_path, f"learning_rate: {args.learning_rate}")
        _write_log(self.log_path, f"evaluation_strategy: {getattr(args, 'evaluation_strategy', None)}")
        _write_log(self.log_path, f"logging_steps: {args.logging_steps}")
        _write_log(self.log_path, "-" * 80)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.last_eval = metrics

            # Write a readable evaluation block, including confusion matrices if present
            ts = datetime.datetime.now().isoformat()
            _write_log(self.log_path, "")
            _write_log(self.log_path, f"[EVAL] time={ts} epoch={state.epoch} step={state.global_step}")
            _write_log(self.log_path, "-" * 80)

            # Pull confusion matrices if present
            env_cm = metrics.get("env_cm")
            edu_cm = metrics.get("edu_cm")
            econ_cm = metrics.get("econ_cm")

            # Write key metrics first (acc + macro f1 + class2 recall/precision)
            def _maybe(name):
                v = metrics.get(name)
                return "NA" if v is None else f"{v:.4f}"

            _write_log(self.log_path, f"env_acc={_maybe('env_acc')} env_f1_macro={_maybe('env_f1_macro')} env_p_c2={_maybe('env_p_c2')} env_r_c2={_maybe('env_r_c2')} env_f1_c2={_maybe('env_f1_c2')}")
            _write_log(self.log_path, f"edu_acc={_maybe('edu_acc')} edu_f1_macro={_maybe('edu_f1_macro')} edu_p_c2={_maybe('edu_p_c2')} edu_r_c2={_maybe('edu_r_c2')} edu_f1_c2={_maybe('edu_f1_c2')}")
            _write_log(self.log_path, f"econ_acc={_maybe('econ_acc')} econ_f1_macro={_maybe('econ_f1_macro')} econ_p_c2={_maybe('econ_p_c2')} econ_r_c2={_maybe('econ_r_c2')} econ_f1_c2={_maybe('econ_f1_c2')}")
            _write_log(self.log_path, "")

            # Write all metrics as JSON (full detail, still readable)
            metrics_clean = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}
            _write_log(self.log_path, "[EVAL METRICS JSON]")
            _write_log(self.log_path, json.dumps(metrics_clean, indent=2))
            _write_log(self.log_path, "")

            # Write confusion matrices in a human-readable table
            if env_cm is not None:
                _write_log(self.log_path, "[ENV CONFUSION MATRIX] (rows=true, cols=pred)")
                _write_log(self.log_path, _fmt_cm(env_cm))
                _write_log(self.log_path, "")
            if edu_cm is not None:
                _write_log(self.log_path, "[EDU CONFUSION MATRIX] (rows=true, cols=pred)")
                _write_log(self.log_path, _fmt_cm(edu_cm))
                _write_log(self.log_path, "")
            if econ_cm is not None:
                _write_log(self.log_path, "[ECON CONFUSION MATRIX] (rows=true, cols=pred)")
                _write_log(self.log_path, _fmt_cm(econ_cm))
                _write_log(self.log_path, "")

            _write_log(self.log_path, "-" * 80)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        # Approx examples seen
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        grad_accum = getattr(args, "gradient_accumulation_steps", 1)
        effective_batch = args.per_device_train_batch_size * world_size * grad_accum
        approx_examples = state.global_step * effective_batch

        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)

        # print to console
        msg = f"[step {state.global_step} | ~{approx_examples} ex] "
        if loss is not None:
            msg += f"train_loss={loss:.4f} "
        if lr is not None:
            msg += f"lr={lr:.2e} "
        print(msg)

        # write training logs to file (every logging step)
        ts = datetime.datetime.now().isoformat()
        line = f"[TRAIN] time={ts} epoch={state.epoch} step={state.global_step} approx_examples={approx_examples}"
        if loss is not None:
            line += f" train_loss={loss:.6f}"
        if lr is not None:
            line += f" lr={lr:.6e}"
        _write_log(self.log_path, line)

    def on_train_end(self, args, state, control, **kwargs):
        _write_log(self.log_path, f"RUN END: {datetime.datetime.now().isoformat()}")
        _write_log(self.log_path, "=" * 80)
        _write_log(self.log_path, "")


# --------------------------------------------------
# Main training function
# --------------------------------------------------
def main(data_csv: str, out_dir: str, model_source: str, epochs: int):
    os.makedirs(out_dir, exist_ok=True)

    # NEW: log file lives inside output dir; include a header timestamp at the top of each run
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, "log.txt")

    df = pd.read_csv(data_csv)

    if "text" in df.columns:
        df["text"] = df["text"].fillna("")
    elif "social_history" in df.columns:
        df["text"] = df["social_history"].fillna("")
    else:
        raise ValueError("Could not find a text column.")

    df = df.rename(columns={
        "sdoh_environment": "environment_label",
        "sdoh_education":   "education_label",
        "sdoh_economics":   "economics_label",
    })

    label_cols = ["environment_label", "education_label", "economics_label"]

    for col in label_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required label column: {col}")
        df[col] = df[col].astype(int)
        if not set(df[col].unique()).issubset({0, 1, 2}):
            raise ValueError(f"Invalid values found in {col}")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["environment_label"],
    )

    # compute class weights from TRAIN split only (per task)
    def compute_class_weights(series, num_classes=3):
        counts = series.value_counts().reindex(range(num_classes), fill_value=0).to_numpy(dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32), counts

    w_env, env_counts   = compute_class_weights(train_df["environment_label"])
    w_edu, edu_counts   = compute_class_weights(train_df["education_label"])
    w_econ, econ_counts = compute_class_weights(train_df["economics_label"])

    print("Env counts:", train_df["environment_label"].value_counts().sort_index().to_dict(), "weights:", w_env.tolist())
    print("Edu counts:", train_df["education_label"].value_counts().sort_index().to_dict(), "weights:", w_edu.tolist())
    print("Econ counts:", train_df["economics_label"].value_counts().sort_index().to_dict(), "weights:", w_econ.tolist())

    # write run config + weights to log.txt (readable)
    _write_log(log_path, "=" * 80)
    _write_log(log_path, f"RUN TAG: {run_ts}")
    _write_log(log_path, f"START: {datetime.datetime.now().isoformat()}")
    _write_log(log_path, f"data_csv: {data_csv}")
    _write_log(log_path, f"model_source: {model_source}")
    _write_log(log_path, f"out_dir: {out_dir}")
    _write_log(log_path, f"epochs: {epochs}")
    _write_log(log_path, f"TRAIN label counts env={env_counts.tolist()} edu={edu_counts.tolist()} econ={econ_counts.tolist()}")
    _write_log(log_path, f"CLASS weights env={w_env.tolist()} edu={w_edu.tolist()} econ={w_econ.tolist()}")
    _write_log(log_path, "-" * 80)

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

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

    model = ClinicalBERT_MTL(
        model_source=model_source,
        num_labels=3,
        local_only=True,
        env_weights=w_env,
        edu_weights=w_edu,
        econ_weights=w_econ,
    )

    # ----------------------------
    # Metrics (per-class + confusion matrices)
    # ----------------------------
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        logits_env, logits_edu, logits_econ = preds
        y_env, y_edu, y_econ = labels

        def to_np(x):
            return x.cpu().numpy() if hasattr(x, "cpu") else x

        logits_env = to_np(logits_env); logits_edu = to_np(logits_edu); logits_econ = to_np(logits_econ)
        y_env = to_np(y_env); y_edu = to_np(y_edu); y_econ = to_np(y_econ)

        p_env = logits_env.argmax(-1)
        p_edu = logits_edu.argmax(-1)
        p_econ = logits_econ.argmax(-1)

        metrics = {}

        def add_metrics(prefix, y_true, y_pred):
            metrics[f"{prefix}_acc"] = float(accuracy_score(y_true, y_pred))

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1, 2], zero_division=0
            )
            for cls in [0, 1, 2]:
                metrics[f"{prefix}_p_c{cls}"]  = float(precision[cls])
                metrics[f"{prefix}_r_c{cls}"]  = float(recall[cls])
                metrics[f"{prefix}_f1_c{cls}"] = float(f1[cls])

            _, _, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics[f"{prefix}_f1_macro"] = float(f1_macro)

            # NEW: confusion matrix (3x3 list for readability)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
            metrics[f"{prefix}_cm"] = cm

        add_metrics("env",  y_env,  p_env)
        add_metrics("edu",  y_edu,  p_edu)
        add_metrics("econ", y_econ, p_econ)

        return metrics

    # ----------------------------
    # Training args with ~1000 example logging
    # ----------------------------
    per_device_train_bs = 8
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = 1
    effective_batch = per_device_train_bs * world_size * grad_accum
    steps_per_1k = max(1, math.ceil(1000 / effective_batch))
    print(f"[Logging] world_size={world_size} effective_batch={effective_batch} -> logging_steps={steps_per_1k} (~1000 examples)")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_bs,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,

        logging_steps=steps_per_1k,
        logging_strategy="steps",
        eval_strategy="epoch",  # changed from evaluation_strategy
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
        callbacks=[ProgressAndFileLoggerCallback(log_path=log_path)],
    )

    trainer.train()

    res = trainer.evaluate()
    print(res)

    trainer.save_model(out_dir)
    print("Saved MTL ClinicalBERT to:", out_dir)

    # Final summary block in log
    _write_log(log_path, "")
    _write_log(log_path, "=" * 80)
    _write_log(log_path, f"FINAL SUMMARY ({datetime.datetime.now().isoformat()})")
    _write_log(log_path, f"Saved model to: {out_dir}")
    _write_log(log_path, "[FINAL EVAL METRICS JSON]")
    # Ensure JSON serializable
    res_clean = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.items()}
    _write_log(log_path, json.dumps(res_clean, indent=2))
    _write_log(log_path, "=" * 80)
    _write_log(log_path, "")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Data/MTL_preprocessed.csv")
    parser.add_argument("--out", type=str, default="models/clinicalbert_sdoh_mtl")
    parser.add_argument("--model", type=str, default="hf_models/Bio_ClinicalBERT")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    main(args.data, args.out, args.model, args.epochs)

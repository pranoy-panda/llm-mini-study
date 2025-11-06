"""
finetune.py

This script demonstrates two *short* fine-tuning runs on a tiny English subset (≈1k tokens):
  1) Full fine-tune (all model params) for a couple of epochs with a tiny LR
  2) LoRA fine-tune using `peft` (r=8 by default)

It uses a short custom training loop (no Trainer), logs:
  - train loss curves (saved as PNG)
  - evaluation perplexity on a small validation split
  - SST (stanford sentiment treebank)-2 zero-shot accuracy (kept for comparability)
  - number of trainable parameters
  - GPU/CPU memory usage (peak where applicable)

The code has matrix/tensor dimension comments at the relevant steps so the math is explicit.

Run examples:
  python finetune.py --model gpt2 --seed 42
"""

import argparse
import math
import os
import time
from typing import List, Tuple
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# For plotting and memory
import matplotlib.pyplot as plt
import psutil

# -------------------- Utilities --------------------

def count_trainable_params(model: torch.nn.Module) -> int:
    """Return count of parameters where requires_grad == True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_bytes(n: int) -> str:
    """Convert bits to human readable bytes and its multiples"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def mem_report(device: str) -> str:
    """Return a short memory report string.

    If CUDA is available we report CUDA peak/reserved/allocated; otherwise CPU RAM.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()
        return f"CUDA allocated={human_bytes(allocated)} reserved={human_bytes(reserved)} peak={human_bytes(peak)}"
    else:
        vm = psutil.virtual_memory()
        return f"CPU used={human_bytes(vm.used)} total={human_bytes(vm.total)}"


# -------------------- dataset creation --------------------
class TextDataset(Dataset):
    """A simple dataset wrapper storing tokenized examples (lists of input_ids).

    Each item is a dict: {"input_ids": List[int]}
    """

    def __init__(self, tokenized_examples: List[List[int]]):
        self.examples = tokenized_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}


def collate_pad(batch: List[dict], pad_id: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Pad a batch of tokenized sequences.

    Returns:
      input_ids: LongTensor, shape (B, L)
      attention_mask: LongTensor, shape (B, L)

    Comments on dims/maths:
      - B = batch size
      - L = maximum sequence length after padding in this batch
      - logits from model will be (B, L, V) where V is vocab size
    """
    seqs = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)  # (B, L)
    attention_mask = (input_ids != pad_id).long()  # (B, L)
    return input_ids, attention_mask


def build_tiny_dataset(tokenizer: AutoTokenizer, split: str = "train", max_total_tokens: int = 1024) -> Tuple[TextDataset, List[dict]]:
    """Build a tiny dataset by accumulating raw texts until we reach ~max_total_tokens tokens.

    We use WikiText (short sentences) to create our dataset. We tokenize each line
    and add it as one example. Stop when total #tokens >= max_total_tokens.
    Returns the dataset and the raw selected examples (for evaluation or debugging).
    """
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tokenized = []
    raw_selected = []
    tot = 0
    for item in raw:
        text = item["text"].strip()
        if len(text) == 0:
            continue
        enc = tokenizer(text, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) == 0:
            continue
        # Trim extremely long lines for our study
        if len(ids) > 128:
            ids = ids[:128]
        tokenized.append(ids)
        raw_selected.append({"text": text})
        tot += len(ids)
        if tot >= max_total_tokens:
            break
    return TextDataset(tokenized), raw_selected


# -------------------- Evaluation functions --------------------

def eval_perplexity(model: torch.nn.Module, dataloader: DataLoader, device: str, tokenizer_pad_id: int) -> float:
    """Compute per-token perplexity on the given dataset. Uses labels=input_ids.

    Note: For causal LM we pass labels=input_ids and let the model compute loss per token.

    Returns perplexity (float).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)  # (B, L)
            attention_mask = attention_mask.to(device)  # (B, L)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # outputs.loss is mean loss over all non-ignored tokens (in nats because model returns loss in natural log)
            n_tokens = int(attention_mask.sum().item())
            batch_loss = float(outputs.loss.item()) * n_tokens
            total_nll += batch_loss
            total_tokens += n_tokens
    per_token_loss = total_nll / total_tokens
    perplexity = math.exp(per_token_loss)
    model.train() # Set model back to train mode
    return perplexity


def sst2_zero_shot_accuracy(model: torch.nn.Module, tokenizer: AutoTokenizer, device: str, num_examples: int = 200) -> float:
    """Compute the simple SST-2 zero-shot scoring accuracy used in the earlier code.

    This function builds prompts and uses label log-prob sums to pick a label.
    """
    # Load small validation subset
    sst = load_dataset("glue", "sst2", split="validation")
    sst = sst.select(range(min(len(sst), num_examples)))

    label_texts = [" positive", " negative"]

    def make_prompt(sentence: str) -> str:
        return f"Sentence: {sentence}\nQuestion: Is the sentiment of the sentence positive or negative?\nAnswer:"

    correct = 0
    total = 0
    BATCH = 8
    items = []
    for ex in sst:
        p = make_prompt(ex["sentence"])
        items.append({"prompt_ids": tokenizer(p, add_special_tokens=False)["input_ids"], "label": ex["label"]})

    # iterate in small batches
    for i in range(0, len(items), BATCH):
        batch = items[i : i + BATCH]
        # collate into 2*B sequences (pos and neg per example)
        seqs = []
        meta = []
        for j, row in enumerate(batch):
            prompt_ids = row["prompt_ids"]
            for label_idx, label_text in enumerate(label_texts):
                label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
                seq = prompt_ids + label_ids
                seqs.append(torch.tensor(seq, dtype=torch.long))
                meta.append((j, label_idx, len(prompt_ids), len(label_ids)))
        input_ids = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)  # (2B, L)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (2B, L, V)
            log_probs = F.log_softmax(logits, dim=-1)

        seq_scores = []
        for seq_idx, (example_idx, label_idx, prompt_len, label_len) in enumerate(meta):
            # label tokens occupy positions prompt_len .. prompt_len+label_len-1
            # prediction for token at position p is logits at position p-1
            lp_sum = 0.0
            for tok_pos in range(label_len):
                abs_pos = prompt_len + tok_pos
                pred_pos = abs_pos - 1
                token_id = input_ids[seq_idx, abs_pos].item()
                lp = log_probs[seq_idx, pred_pos, token_id].item()
                lp_sum += lp
            seq_scores.append(lp_sum)

        # compare pairs
        for j in range(len(batch)):
            score_pos = seq_scores[2 * j + 0]
            score_neg = seq_scores[2 * j + 1]
            pred = 1 if score_pos > score_neg else 0
            if pred == batch[j]["label"]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


# -------------------- Training loop --------------------

def train_short_run(
    base_model_name: str,
    tokenizer: AutoTokenizer,
    device: str,
    tiny_dataset: TextDataset,
    val_loader: DataLoader,
    run_name: str = "full",
    use_lora: bool = False,
    lora_r: int = 8,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    save_dir: str = "outputs",
):
    """Train a model for a small number of steps and return diagnostics.

    - use_lora: if True, wraps the model in PEFT LoRA with r=lora_r
    - returns dictionary with logs and filepaths
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    model.train()

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # Optionally apply LoRA
    if use_lora:
        # LoRA config for full attention/query/key/value projection matrices
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=["c_attn", "q_proj", "k_proj", "v_proj"],
            # The `target_modules` names depend on model architecture; GPT-2 uses c_attn
            # For models with q_proj/k_proj/v_proj naming to be adjusted accordingly.
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("Applied LoRA; only LoRA params require grad.")

    # Count trainable parameters
    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())

    print(f"Run={run_name} use_lora={use_lora} trainable_params={trainable} total_params={total}")

    # DataLoader
    def collate_fn(batch):
        return collate_pad(batch, pad_id=tokenizer.pad_token_id)

    loader = DataLoader(tiny_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer over parameters with requires_grad
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Reset peak memory counters if CUDA
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    losses = []
    perplexities = []
    iters = 0
    start_train_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        for input_ids, attention_mask in loader:
            # input_ids: (B, L)
            # attention_mask: (B, L)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            
            # outputs.loss: cross-entropy loss (specifically nn.CrossEntropyLoss), applied tokenwise over the vocabulary logits.
            # Also, its the mean loss over non-masked tokens (scalar), i.e., scalar over tokens
            loss = outputs.loss
            # Backprop
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optim.step()
            optim.zero_grad()

            # Logging
            # Number of non-pad tokens in batch:
            n_tokens = int(attention_mask.sum().item())
            batch_loss = float(loss.item()) * n_tokens
            epoch_loss += batch_loss
            epoch_tokens += n_tokens
            iters += 1
            losses.append(float(loss.item()))

            if iters % 10 == 0:
                print(f"{run_name} epoch={epoch} iter={iters} loss={loss.item():.4f} token_loss={batch_loss/n_tokens:.4f} mem={mem_report(device)}")

        # End epoch
        per_tok_loss = epoch_loss / epoch_tokens
        print(f"{run_name} end epoch={epoch} per-token-loss(nats)={per_tok_loss:.4f} perp={math.exp(per_tok_loss):.4f}")

        # Evaluate perplexity on the validation set at the end of the epoch
        epoch_perplexity = eval_perplexity(model, val_loader, device, tokenizer.pad_token_id)
        perplexities.append(epoch_perplexity)
        print(f"{run_name} end epoch={epoch} validation_perplexity={epoch_perplexity:.4f}")

    total_train_time = time.time() - start_train_time

    # Peak memory usage
    mem = mem_report(device)

    # Save LoRA adapter if used
    if use_lora:
        # peft models support .save_pretrained
        peft_out = os.path.join(save_dir, f"{run_name}_lora")
        model.save_pretrained(peft_out)
        print(f"Saved LoRA weights to {peft_out}")

    # Save training loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("train_loss (mean over batch tokens)")
    plt.title(f"Train loss curve ({run_name})")
    plot_path = os.path.join(save_dir, f"train_loss_{run_name}.png")
    plt.savefig(plot_path)
    plt.close()

    return {
        "model": model,
        "trainable_params": trainable,
        "total_params": total,
        "loss_curve_path": plot_path,
        "loss_values": losses,
        "train_time_s": total_train_time,
        "mem_report": mem,
        "perplexities": perplexities,
    }


# -------------------- Main script --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024, help="~total tokens to assemble for training (tiny demo)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build tiny dataset (~1k tokens total) from WikiText train split
    tiny_dataset, raw_selected = build_tiny_dataset(tokenizer, split="train", max_total_tokens=args.max_tokens)
    print(f"Built tiny dataset with {len(tiny_dataset)} examples (~{args.max_tokens} tokens target)")

    # Quick dataloader for eval purposes (use validation split)
    val_dataset, _ = build_tiny_dataset(tokenizer, split="validation", max_total_tokens=512)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=lambda b: collate_pad(b, pad_id=tokenizer.pad_token_id))

    # ---------- Run A: Full fine-tune (all parameters) ----------
    print("Starting full fine-tune (all params)...")
    full_res = train_short_run(
        base_model_name=args.model,
        tokenizer=tokenizer,
        device=device,
        tiny_dataset=tiny_dataset,
        val_loader=val_loader,
        run_name="full",
        use_lora=False,
        epochs=2,
        batch_size=4,
        lr=1e-5,  # tiny LR for full fine-tune
        save_dir="outputs",
    )

    # Evaluate full model perplexity and SST-2 acc
    print("Evaluating full model on tiny validation set and SST-2 zero-shot...")
    full_model = full_res["model"]
    full_model.eval()
    perplex_full = eval_perplexity(full_model, val_loader, device, tokenizer.pad_token_id)
    sst_acc_full = sst2_zero_shot_accuracy(full_model, tokenizer, device, num_examples=200)

    print(f"Full: trainable_params={full_res['trainable_params']} total_params={full_res['total_params']}")
    print(f"Full: perplexity on tiny val = {perplex_full:.4f}")
    print(f"Full: SST-2 zero-shot acc = {sst_acc_full:.2%}")
    print(f"Full memory: {full_res['mem_report']}")

    # ---------- Run B: LoRA fine-tune ----------
    print("Starting LoRA fine-tune (r=8)...")
    lora_res = train_short_run(
        base_model_name=args.model,
        tokenizer=tokenizer,
        device=device,
        tiny_dataset=tiny_dataset,
        val_loader=val_loader,
        run_name="lora",
        use_lora=True,
        lora_r=8,
        epochs=2,
        batch_size=4,
        lr=1e-4,  # slightly larger LR for LoRA
        save_dir="outputs",
    )

    print("Evaluating LoRA model on tiny validation set and SST-2 zero-shot...")
    lora_model = lora_res["model"]
    lora_model.eval()
    perplex_lora = eval_perplexity(lora_model, val_loader, device, tokenizer.pad_token_id)
    sst_acc_lora = sst2_zero_shot_accuracy(lora_model, tokenizer, device, num_examples=200)

    print(f"LoRA: trainable_params={lora_res['trainable_params']} total_params={lora_res['total_params']}")
    print(f"LoRA: perplexity on tiny val = {perplex_lora:.4f}")
    print(f"LoRA: SST-2 zero-shot acc = {sst_acc_lora:.2%}")
    print(f"LoRA memory: {lora_res['mem_report']}")

    # Plot perplexity comparison
    plt.figure()
    plt.plot(full_res["perplexities"], label="Full Fine-Tune")
    plt.plot(lora_res["perplexities"], label="LoRA (r=8)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Perplexity")
    plt.title("Validation Perplexity Across Epochs")
    plt.legend()
    plt.grid(True)
    perplexity_plot_path = os.path.join("outputs", "perplexity_comparison.png")
    plt.savefig(perplexity_plot_path)
    plt.close()

    # Summarize and save a small text report
    summary = {
        "full": {
            "trainable_params": full_res["trainable_params"],
            "total_params": full_res["total_params"],
            "perplexity": perplex_full,
            "sst2_acc": sst_acc_full,
            "loss_curve": full_res["loss_curve_path"],
            "mem": full_res["mem_report"],
            "epoch_perplexities": full_res["perplexities"],
        },
        "lora": {
            "trainable_params": lora_res["trainable_params"],
            "total_params": lora_res["total_params"],
            "perplexity": perplex_lora,
            "sst2_acc": sst_acc_lora,
            "loss_curve": lora_res["loss_curve_path"],
            "mem": lora_res["mem_report"],
            "epoch_perplexities": lora_res["perplexities"],
        },
        "perplexity_plot": perplexity_plot_path,
    }

    with open("outputs/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Results saved to outputs/ (loss curve PNGs, perplexity plot, and outputs/summary.json)")

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from src.finetune import (
    build_tiny_dataset,
    collate_pad,
    train_short_run,
    eval_perplexity,
    TextDataset
)

def attention_ablation_hook(module, input, output):
    """
    A forward hook to ablate attention scores. Note: A pytorch `hook` is a callback function that we use with nn.Module. A forward hook is executed when the model does its forward pass.
    
    This hook sets the attention scores to zeros before the softmax. By setting the attention scores to zero before the softmax, we force the softmax to output a uniform distribution: softmax(zeros) = [1/N, 1/N, ..., 1/N] where N is the sequence length.
Thus to compute the representation for a given token, the model is forced to take a simple, unweighted average of all the other token representations in the sequence.
    """
    # The output of the attention module before softmax is a tuple,
    # where the first element is the attention scores.
    # Dimensions: (batch_size, num_heads, seq_len, seq_len)
    attention_scores = output[0]
    
    # Set attention scores to zeros
    ablated_scores = torch.zeros_like(attention_scores[0])
    
    # Return the modified scores
    return (ablated_scores,) + output[1:]

# Apply the hook to the attention modules of the model
def apply_attention_ablation(model):
    for layer in model.transformer.h:
        layer.attn.register_forward_hook(attention_ablation_hook)

# Now, let's integrate this into a training run.
# We'll create a new training function for this experiment.

def train_ablation_run(
    base_model_name: str,
    tokenizer: AutoTokenizer,
    device: str,
    tiny_dataset: TextDataset,
    val_loader: DataLoader,
    run_name: str = "ablation",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    save_dir: str = "outputs",
):
    """
    Train a model with attention ablation and return diagnostics.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
    
    # Apply the attention ablation hook
    apply_attention_ablation(model)
    
    model.train()

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # DataLoader
    def collate_fn(batch):
        return collate_pad(batch, pad_id=tokenizer.pad_token_id)

    loader = DataLoader(tiny_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    perplexities = []
    iters = 0

    for epoch in range(epochs):
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            losses.append(float(loss.item()))
            iters += 1

            if iters % 10 == 0:
                print(f"{run_name} epoch={epoch} iter={iters} loss={loss.item():.4f}")

        # Evaluate perplexity
        epoch_perplexity = eval_perplexity(model, val_loader, device, tokenizer.pad_token_id)
        perplexities.append(epoch_perplexity)
        print(f"{run_name} end epoch={epoch} validation_perplexity={epoch_perplexity:.4f}")

    # Save training loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("train_loss")
    plt.title(f"Train loss curve ({run_name})")
    plot_path = os.path.join(save_dir, f"train_loss_{run_name}.png")
    plt.savefig(plot_path)
    plt.close()

    return {
        "loss_values": losses,
        "perplexities": perplexities,
    }

# In main script or notebook, we would run this like:
# ablation_res = train_ablation_run(...)


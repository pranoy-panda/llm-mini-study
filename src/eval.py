import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math

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
    
    
def eval_classification_accuracy(model, tokenizer, test_dataset, device, batch_size=32):
    """
    Evaluates the accuracy of a fine-tuned sequence classification model.

    Args:
        model (torch.nn.Module): The transformer model with a classification head.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        test_dataset (datasets.Dataset): The evaluation dataset.
        device (str): The device to evaluate on ('cuda' or 'cpu').
        batch_size (int): Evaluation batch size.

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    # Set up DataLoader
    def collate_fn(batch):
        premises = [item['premise'] for item in batch]
        hypotheses = [item['hypothesis'] for item in batch]
        labels = [item['label'] for item in batch]
    
        # Format the input as "premise [SEP] hypothesis"
        inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        return inputs

    loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Update counts
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples if total_samples > 0 else 0.0

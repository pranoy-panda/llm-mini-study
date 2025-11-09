import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
        texts = [item['review_body'] for item in batch]
        labels = [item['label'] for item in batch]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
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

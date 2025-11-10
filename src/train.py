import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_classifier(model, tokenizer, train_dataset, device, epochs=3, lr=2e-5, batch_size=16):
    """
    Fine-tunes a Transformers model for sequence classification.

    Args:
        model (torch.nn.Module): The transformer model with a classification head.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        train_dataset (datasets.Dataset): The training dataset.
        device (str): The device to train on ('cuda' or 'cpu').
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Training batch size.

    Returns:
        torch.nn.Module: The fine-tuned model.
    """
    # Set up DataLoader
    def nli_collate_fn(batch):
        premises = [item['premise'] for item in batch]
        hypotheses = [item['hypothesis'] for item in batch]
        labels = [item['label'] for item in batch]
        # Format the input as "premise [SEP] hypothesis"
        inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels, dtype=torch.long)
        return inputs

    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=nli_collate_fn, shuffle=True)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
    return model

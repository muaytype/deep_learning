import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from lab9_transformer import Seq2SeqTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware acquired: {device}")

print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("abisee/cnn_dailymail", "1.0.0", split="train[:5000]")

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
vocab_size = len(tokenizer)
pad_idx = tokenizer.pad_token_id

def preprocess(examples):
    inputs = tokenizer(examples['article'], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(examples['highlights'], truncation=True, padding="max_length", max_length=128)
    return {"src": inputs["input_ids"], "tgt": targets["input_ids"]}

tokenized_data = dataset.map(preprocess, batched=True).remove_columns(["article", "highlights", "id"])
tokenized_data.set_format(type="torch", columns=["src", "tgt"])

dataloader = DataLoader(tokenized_data, batch_size=32, shuffle=True)

model = Seq2SeqTransformer(vocab_size=vocab_size).to(device, dtype=torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

epochs = 3
model.train()

for epoch in range(epochs):
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        optimizer.zero_grad()

        decoder_input = tgt[:, :-1]
        seq_len = decoder_input.size(1)

        tgt_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))

        output_probs, _ = model(src, decoder_input, tgt_mask=tgt_mask)

        loss = criterion(output_probs.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f} ---")

torch.save(model.state_dict(), "transformer_summarizer.pth")
print("Training complete. Model weights saved.")

import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dataset_file = "alice.txt"
if not os.path.exists(dataset_file):
    urllib.request.urlretrieve("https://www.gutenberg.org/files/11/11-0.txt", dataset_file)

with open(dataset_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

data_tensor = torch.tensor([char_to_int[ch] for ch in text], dtype=torch.int64)

batch_size = 64
seq_length = 100

def get_batch(data_tensor, batch_size, seq_length):
    max_index = len(data_tensor) - seq_length - 1
    ix = torch.randint(0, max_index, (batch_size,))

    x = torch.stack([data_tensor[i : i+seq_length] for i in ix])
    y = torch.stack([data_tensor[i+1 : i+seq_length+1] for i in ix])
    return x, y

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden

model = CharRNN(vocab_size=vocab_size, embed_size=128, hidden_size=256)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 2000
model.train()

for epoch in range(epochs):
    X, Y = get_batch(data_tensor, batch_size, seq_length)
    X, Y = X.to(device), Y.to(device)

    logits, hidden = model(X, hidden=None)

    loss = criterion(logits.view(-1, vocab_size), Y.view(-1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")

def generate_text(model, start_string, generation_length, temperature=0.8):
    model.eval()

    input_eval = torch.tensor([char_to_int[s] for s in start_string], dtype=torch.int64).unsqueeze(0).to(device)
    generated_text = start_string
    hidden = None

    with torch.no_grad():
        for _ in range(generation_length):
            logits, hidden = model(input_eval, hidden)

            next_char_logits = logits[0, -1, :] / temperature
            probabilities = F.softmax(next_char_logits, dim=0)

            next_index = torch.multinomial(probabilities, num_samples=1).item()
            generated_text += int_to_char[next_index]

            input_eval = torch.tensor([[next_index]], dtype=torch.int64).to(device)

    return generated_text

print("\n--- Final Generated Output ---")
print(generate_text(model, start_string="Alice ", generation_length=400, temperature=0.7))

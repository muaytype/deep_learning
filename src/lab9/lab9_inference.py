import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from lab9_transformer import Seq2SeqTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing inference on: {device}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
vocab_size = len(tokenizer)

model = Seq2SeqTransformer(vocab_size=vocab_size).to(device, dtype=torch.bfloat16)

try:
    model.load_state_dict(torch.load("transformer_summarizer.pth", map_location=device))
    print("Successfully loaded trained weights.")
except FileNotFoundError:
    print("CRITICAL: 'transformer_summarizer.pth' not found. Did the training script finish?")
    exit(1)

model.eval()

def generate_summary(model, article_text, max_len=64, temperature=0.8):
    src_tokens = tokenizer(article_text, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)

    decoder_input = torch.tensor([[tokenizer.cls_token_id]], device=device)

    with torch.no_grad():
        for i in range(max_len):
            seq_len = decoder_input.size(1)
            tgt_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))

            output_logits, _ = model(src_tokens, decoder_input, tgt_mask=tgt_mask)

            next_word_logits = output_logits[:, -1, :]

            scaled_logits = next_word_logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)

            next_word_token = torch.multinomial(probabilities, num_samples=1)

            decoder_input = torch.cat([decoder_input, next_word_token], dim=1)

            if next_word_token.item() == tokenizer.sep_token_id:
                break

    generated_integers = decoder_input.squeeze().tolist()
    return tokenizer.decode(generated_integers, skip_special_tokens=True)

if __name__ == "__main__":
    print("\nDownloading CNN/DailyMail Test Split...")
    test_dataset = load_dataset("abisee/cnn_dailymail", "1.0.0", split="test")

    sample_data = test_dataset[0]
    raw_article = sample_data["article"]
    ground_truth = sample_data["highlights"]

    print("\n--- ORIGINAL ARTICLE (Truncated) ---")
    print(raw_article[:600] + "...\n")

    print("--- GROUND TRUTH SUMMARY (Human Written) ---")
    print(ground_truth + "\n")

    print("--- MODEL GENERATED SUMMARY (Temperature: 0.8) ---")
    generated_text = generate_summary(model, raw_article, temperature=0.8)
    print(generated_text + "\n")

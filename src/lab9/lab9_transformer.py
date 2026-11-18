import torch
import torch.nn as nn
import torch.nn.functional as F

from lab9_encoder import EncoderLayer, PositionalEncoding
from lab9_decoder import DecoderLayer

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048, num_layers=6, dropout_rate=0.1):
        super(Seq2SeqTransformer, self).__init__()

        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ])

        self.generator = Generator(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.pos_encoder(self.src_embed(src))

        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        dec_output = self.pos_encoder(self.tgt_embed(tgt))

        for layer in self.decoder_layers:
            dec_output, attn_weights = layer(dec_output, enc_output, src_mask, tgt_mask)

        final_output = self.generator(dec_output)

        return final_output, attn_weights

if __name__ == "__main__":
    batch_size = 32
    src_len = 512
    tgt_len = 128
    vocab_size = 119547

    dummy_src = torch.randint(0, vocab_size, (batch_size, src_len))
    dummy_tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).unsqueeze(0)
    tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))

    model = Seq2SeqTransformer(vocab_size=vocab_size, num_layers=2)

    output_probs, _ = model(dummy_src, dummy_tgt, tgt_mask=tgt_mask)

    print(f"Final Generator Output Shape: {output_probs.shape}")
    

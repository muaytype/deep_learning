import torch
import torch.nn as nn
import numpy as np
from lab9_encoder import PositionwiseFeedForward
from lab9_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.enc_attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        attn_out, _ = self.attn_layer(x, x, x, tgt_mask)
        x = self.attn_layer_norm(x + self.dropout(attn_out))

        enc_attn_out, attention_weights = self.enc_attn_layer(x, memory, memory, src_mask)
        x = self.enc_attn_layer_norm(x + self.dropout(enc_attn_out))

        ff_out = self.ff_layer(x)
        x = self.ff_layer_norm(x + self.dropout(ff_out))

        return x, attention_weights

if __name__ == "__main__":
    batch_size = 32
    src_seq_length = 512
    tgt_seq_length = 128
    d_model = 512
    n_heads = 8
    d_ff = 2048

    encoder_memory = torch.randn(batch_size, src_seq_length, d_model)

    decoder_input = torch.randn(batch_size, tgt_seq_length, d_model)

    tgt_mask = torch.tril(torch.ones((tgt_seq_length, tgt_seq_length)))

    tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))

    tgt_mask = tgt_mask.unsqueeze(0)

    decoder_layer = DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    output, cross_attn_weights = decoder_layer(x=decoder_input, memory=encoder_memory, tgt_mask=tgt_mask)

    print(f"Encoder Memory Shape: {encoder_memory.shape}")
    print(f"Decoder Input Shape: {decoder_input.shape}")
    print(f"Decoder Output Shape: {output.shape}")
    print(f"Cross-Attention Weights Shape: {cross_attn_weights.shape}")

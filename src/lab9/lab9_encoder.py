import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lab9_attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn_layer(x, x, x, mask)
        x = self.attn_layer_norm(x + self.dropout(attn_out))

        ff_out = self.ff_layer(x)
        x = self.ff_layer_norm(x + self.dropout(ff_out))

        return x

if __name__ == "__main__":
    batch_size = 32
    seq_length = 128
    d_model = 512
    n_heads = 8
    d_ff = 2048

    dummy_input = torch.randn(batch_size, seq_length, d_model)
    print(f"Input Matrix Shape: {dummy_input.shape}")

    encoder_layer = EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff)

    pos_encoder = PositionalEncoding(d_model=d_model)
    positional_input = pos_encoder(dummy_input)

    output = encoder_layer(positional_input)

    print(f"Encoder Output Shape: {output.shape}")

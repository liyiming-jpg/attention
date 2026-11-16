import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        q = self.q_proj(q).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attn_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
        max_len: int = 32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.output_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_tgt_mask(self, tgt_len: int):
        mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.dropout(self.positional_encoding(self.embedding(src)))
        tgt_emb = self.dropout(self.positional_encoding(self.embedding(tgt)))

        for layer in self.encoder_layers:
            src_emb = layer(src_emb)

        tgt_mask = self.make_tgt_mask(tgt.size(1)).to(tgt.device)
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, tgt_mask)

        return self.output_head(tgt_emb)


def build_dataset(num_samples: int = 512, seq_len: int = 6, vocab_size: int = 8):
    samples = []
    for _ in range(num_samples):
        length = torch.randint(1, seq_len + 1, (1,)).item()
        src = torch.randint(1, vocab_size, (length,))
        tgt = src.clone()
        samples.append((src, tgt))
    return samples


def collate_batch(batch):
    max_len = max(len(src) for src, _ in batch)
    src_batch = []
    tgt_batch = []
    for src, tgt in batch:
        pad_len = max_len - len(src)
        src_pad = F.pad(src, (0, pad_len), value=0)
        tgt_pad = F.pad(tgt, (0, pad_len), value=0)
        src_batch.append(src_pad)
        tgt_batch.append(tgt_pad)
    return torch.stack(src_batch), torch.stack(tgt_batch)


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(vocab_size=9, d_model=32, n_heads=4, n_layers=2, d_ff=64, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    dataset = build_dataset(num_samples=1024, seq_len=6, vocab_size=8)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_batch,
    )

    model.train()
    for epoch in range(30):
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()

            logits = model(src, tgt)
            logits = logits.view(-1, logits.size(-1))
            targets = tgt.view(-1)

            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"epoch={epoch + 1:02d} loss={total_loss / max(len(loader), 1):.4f}")

    model.eval()
    with torch.no_grad():
        sample_src = torch.tensor([[1, 2, 3, 4], [5, 1, 7, 2]], device=device)
        sample_tgt = torch.tensor([[1, 2, 3, 4], [5, 1, 7, 2]], device=device)
        logits = model(sample_src, sample_tgt)
        pred = logits.argmax(dim=-1)
        print("sample predictions:")
        print(pred)


def main():
    train_model()


if __name__ == "__main__":
    main()


__all__ = [
    "PositionalEncoding",
    "MultiHeadAttention",
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "Transformer",
    "train_model",
    "main",
]

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

from .utils import init_weight


class Attention(nn.Module):
    def __init__(
        self,
        emb_dim,
        max_seq_len,
        num_heads=12,
        dropout=0.1
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * self.num_heads == self.emb_dim, "Invalid (emb_dim, num_heads) pair."

        self.max_seq_len = max_seq_len

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.wq = nn.Linear(in_features=emb_dim, out_features=emb_dim * num_heads)
        self.wk = nn.Linear(in_features=emb_dim, out_features=emb_dim * num_heads)
        self.wv = nn.Linear(in_features=emb_dim, out_features=emb_dim * num_heads)
        self.fc = nn.Linear(in_features=emb_dim, out_features=emb_dim)

        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x):
        """
        funct: (N, seq_len, emb_dim * num_heads) -> (N, num_heads, seq_len, emb_dim)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.emb_dim)

        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        """
        funct: (N, num_heads, seq_len, emb_dim) -> (N, seq_len, emb_dim * num_heads)
        """
        batch_size, num_heads, seq_len, emb_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.view(batch_size, seq_len, num_heads * emb_dim)

    def attention(self, q, k, v, attention_mask=None):
        """
        funct: calculate attention score ...
            Softmax((Q @ K^T) / sqrt(d_k)) @ V
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if attention_mask is not None:
            scores += attention_mask
        scores = nn.Softmax(dim=-1)(scores)
        scores = self.dropout(scores)
        scores = torch.matmuL(scores, v)

        return scores

    def forward(self, x, attention_mask=None):
        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        attention_scores = self.attention(q, k, v, attention_mask)
        attention_scores = self.merge_heads(attention_scores)
        attention_scores = self.dropout(self.fc(attention_scores))

        return attention_scores


class FeedForward(nn.Module):
    def __init__(self, emb_dim=768, dropout=0.1):
        super().__init__()
        inner_state_dim = emb_dim * 4

        self.fc1 = nn.Linear(in_features=emb_dim, out_features=inner_state_dim)
        self.fc2 = nn.Linear(in_features=inner_state_dim, out_features=emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.dropout(self.fc2(self.gelu(self.fc1(x))))
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        max_seq_len,
        num_heads=12,
        dropout=0.1
    ):
        super().__init__()
        self.attention = Attention(
            emb_dim=emb_dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            dropout=dropout
        )
        self.feedforward = FeedForward(
            emb_dim=emb_dim,
            dropout=dropout
        )
        self.layer_norm1 = LayerNorm(normalized_shape=emb_dim, eps=1e-5)
        self.layer_norm2 = LayerNorm(normalized_shape=emb_dim, eps=1e-5)

    def forward(self, x, attention_mask):
        x += self.attention(self.layer_norm1(x, attention_mask))
        x += self.feedforward(self.layer_norm2(x))

        return x


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        max_seq_len,
        num_heads=12,
        num_layers=12,
        dropout=0.1
    ):
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, emb_dim)
        self.pos_embs = nn.Embedding(max_seq_len, emb_dim)

        blocks = [
            Decoder(
                emb_dim=emb_dim,
                max_seq_len=max_seq_len,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        self.dropout = nn.Dropout(p=dropout)

        self.decoder = nn.Sequential(*blocks)
        self.layer_norm = LayerNorm(normalized_shape=emb_dim, eps=1e-5)
        self.fc = nn.Linear(in_features=emb_dim, out_features=vocab_size, bias=False)

        self.apply(init_weight)

    def get_input_embeddings(self):
        return self.token_embs

    def set_input_embeddings(self, embeddings):
        self.token_embs = embeddings

    def forward(self, x):
        """
        Arg:
            x: Tuple(input_ids, attention_mask)
                e.g., tokenizer("날씨가 매우 화창하다.")
                        input_ids: [34018, 9655, 9192, 8344, 19572]
                        attention_mask: [1, 1, 1, 1, 1]
        """
        input_ids, attention_mask = x
        
        pos_ids = torch.arange(0, input_ids.size(-1)).unsqueeze(0)
        input_embs = self.token_embs(input_ids) + self.pos_embs(pos_ids)
        input_embs = self.dropout(input_embs)

        if attention_mask is not None:
            attention_mask = attention_mask.view(input_ids.size(0), 1, 1, -1) # (N, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0

        logits = self.fc(self.layer_norm(self.decoder(input_embs, attention_mask)))

        return logits
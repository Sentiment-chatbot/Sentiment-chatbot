import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import init_weight
from .option import Prompts, Prompt_idx2word

class Attention(nn.Module):
    """ Multi-head attention """
    def __init__(
        self,
        emb_dim,
        max_seq_len,
        num_heads,
        dropout
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * self.num_heads == self.emb_dim, "Invalid (emb_dim, num_heads) pair."

        self.max_seq_len = max_seq_len

        # biases are used for making causal attention masks
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.uint8)).view(
                1, 1, max_seq_len, max_seq_len
            )
        )
        self.register_buffer("masking_value", torch.tensor(-1e4))

        self.wqkv = nn.Linear(in_features=emb_dim, out_features=emb_dim * 3)
        self.fc = nn.Linear(in_features=emb_dim, out_features=emb_dim)

        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x):
        """
        Args:
            x: (N, seq_len, emb_dim) == (N, seq_len, head_dim * num_heads)
        Funct: 
            (N, seq_len, head_dim * num_heads) -> (N, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)

        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        """
        Args:
            (N, num_heads, seq_len, head_dim) -> (N, seq_len, head_dim * num_heads)
        Funct:
            (N, num_heads, seq_len, head_dim) -> (N, seq_len, head_dim * num_heads) == (N, seq_len, emb_dim)
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.view(batch_size, seq_len, num_heads * head_dim)

    def attention(self, q, k, v, attention_mask=None):
        """
        Args:
            q: (N, num_heads, seq_len, head_dim)
            k: (N, num_heads, seq_len, head_dim)
            v: (N, num_heads, seq_len, head_dim)
            attention_mask: (N, 1, 1, seq_len) if not None
        Funct: calculate attention score
            Softmax((Q @ K^T) / sqrt(d_k)) @ V
        """
        
        scores = torch.matmul(
            q, k.transpose(-2, -1)) / math.sqrt(k.size(-1)
        )  # (N, num_heads, seq_len, seq_len)
        
        # Masked self-attention
        causal_mask = self.bias[:, :, :q.size(-2), :q.size(-2)].bool()
        scores = torch.where(causal_mask, scores, self.masking_value)

        if attention_mask is not None:
            scores += attention_mask # broadcasting => (N, num_heads, seq_len, seq_len)
        scores = nn.Softmax(dim=-1)(scores) # (N, num_heads, seq_len, seq_len)
        scores = self.dropout(scores) # (N, num_heads, seq_len, seq_len)
        scores = torch.matmul(scores, v) # (N, num_heads, seq_len, head_dim)

        return scores # (N, num_heads, seq_len, head_dim)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (N, seq_len, emb_dim)
            attention_mask: (N, 1, 1, seq_len) if not None
        """

        x = self.wqkv(x)
        q, k, v = x.split(self.emb_dim, dim=-1)
        q, k, v = map(lambda m: self.split_heads(m), (q, k, v))

        attention_scores = self.attention(q, k, v, attention_mask) # (N, num_heads, seq_len, head_dim)
        attention_scores = self.merge_heads(attention_scores) # (N, seq_len, head_dim * num_heads)
        attention_scores = self.dropout(self.fc(attention_scores)) # (N, seq_len, emb_dim)

        return attention_scores # (N, seq_len, emb_dim)


class FeedForward(nn.Module):
    """ Feed-forward layer of decoder(Transformer) module """
    def __init__(self, emb_dim, dropout):
        super().__init__()
        inner_state_dim = emb_dim * 4

        self.fc1 = nn.Linear(in_features=emb_dim, out_features=inner_state_dim)
        self.fc2 = nn.Linear(in_features=inner_state_dim, out_features=emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (N, seq_len, emb_dim)
        """
        out = self.gelu(self.fc1(x)) # (N, seq_len, inner_state_dim)
        out = self.dropout(self.fc2(out)) # (N, seq_len, emb_dim)

        return out # (N, seq_len, emb_dim)


class Decoder(nn.Module):
    """ Transformer decoder block """
    def __init__(
        self,
        emb_dim,
        max_seq_len,
        num_heads,
        dropout
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

    def forward(self, x, attention_mask=None):
        """
        Args:
            x:(N, seq_len, emb_dim)
            attention_mask:(N, 1, 1, seq_len)
        """
        x = x + self.attention(self.layer_norm1(x), attention_mask) # (N, seq_len, emb_dim)
        x = x + self.feedforward(self.layer_norm2(x)) # (N, seq_len, emb_dim)

        return x # (N, seq_len, emb_dim)


class GPT2Model(nn.Module):
    """ 
        GPT-2 with last fully connected layer which is used for LM head 
    """
    def __init__(
        self,
        vocab_size,
        emb_dim,
        max_seq_len,
        num_heads,
        num_layers,
        dropout,
        device
    ):
        super().__init__()
        self.token_embs = nn.Embedding(vocab_size, emb_dim)
        self.pos_embs = nn.Embedding(max_seq_len, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.decoders = nn.ModuleList(
            Decoder(
                emb_dim=emb_dim,
                max_seq_len=max_seq_len,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        )
        self.layer_norm = LayerNorm(normalized_shape=emb_dim, eps=1e-5)
        self.fc = nn.Linear(in_features=emb_dim, out_features=vocab_size, bias=False)

        self.device = device

        self.apply(init_weight) # Initializes weights of all inner layers

    def get_input_embeddings(self):
        return self.token_embs

    def set_input_embeddings(self, embeddings):
        self.token_embs = embeddings

    def forward(self, input_ids, attention_ids=None):
        """
        Args:
            input_ids: (N, seq_len)
            attention_ids: (N, seq_len)

            Example 1) Single sentence
                tokenizer("날씨가 굉장히 화창하네?")
                input_ids: 
                    [34018, 40052, 8168, 8811, 9192, 8344, 8702, 7098, 406]
                attention_ids:
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]
                decoded:
                    '▁날씨가', '▁굉', '장', '히', '▁화', '창', '하', '네', '?'
                
            Example 2) Batched sentences
                tokenizer(["날씨가 굉장히 화창하네?", "정말 그렇네요."])
                input_ids: [
                    [34018, 40052, 8168, 8811, 9192, 8344, 8702, 7098, 406],
                    [29205, 11928, 7098, 25856, 3, 3, 3, 3, 3]
                ]
                attention_ids: [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0]
                ]
                decoded: 
                    '▁날씨가', '▁굉', '장', '히', '▁화', '창', '하', '네', '?'
                    '▁정말', '▁그렇', '네', '요.', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'
            
            Since we also used special tokens, real input_ids is like below.
            <s> <sp1> '▁날씨가', '▁굉', '장', '히', '▁화', '창', '하', '네', '?' <sp2> '▁정말', '▁그렇', '네', '요.' </s>
        Return:
            logits: (N, seq_len, vocab_size)
            Each of the logits indicate a confidence of each word for each position
        """
        pos_ids = torch.arange(0, input_ids.size(-1), device=self.device).unsqueeze(0) # (1, seq_len)
        input_embs = self.token_embs(input_ids) + self.pos_embs(pos_ids) # (N, seq_len, emb_dim)
        input_embs = self.dropout(input_embs) # (N, seq_len, emb_dim)

        attention_mask = None
        if attention_ids is not None:
            attention_ids = attention_ids.view(input_ids.size(0), 1, 1, -1) # (N, 1, 1, seq_len)
            attention_mask = (1.0 - attention_ids) * -1e4 # (N, 1, 1, seq_len)

        logits = input_embs # (N, seq_len, emb_dim)
        for decoder in self.decoders:
            logits = decoder(logits, attention_mask) # (N, seq_len, emb_dim)
        logits = self.fc(self.layer_norm(logits)) # (N, seq_len, vocab_size)

        return logits


class EmoClassifier(nn.Module):
    """ 
        Emotion classifier with (bidirectional) LSTM
    """
    def __init__(self, num_layers, emb_dim, hidden_dim, num_classes, dropout, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=emb_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(
            in_features=hidden_dim * (2 if bidirectional else 1),
            out_features=num_classes
        )

    def forward(self, input_embed, seq_len): 
        """
        Args:
            input_embed: (N, seq_len, emb_dim)
            seq_len: (N) (length of each sequence, excluding padded part)

        Return:
            logits: (N, C(6))
            Emotion logits
        """
        packed_embed = pack_padded_sequence(input_embed, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed_embed)
        out, _ = pad_packed_sequence(out, batch_first=True) # (N, seq_len, hidden_dim)
        out = out[:,-1,:]  # (N, hidden_dim)
        logits = self.fc(self.dropout(out)) # (N, hidden_dim) -> (N, C)

        return logits


class EmoGPT2(nn.Module):
    """ 
        GPT2Model with emotion classifer & emotion prompt
    """
    def __init__(
        self,
        emb_dim,
        max_seq_len,
        num_heads,
        num_layers,
        dropout,
        emo_classifier_conf,
        tokenizer,
        device
    ):
        super().__init__()
        self.gpt = GPT2Model(
            len(tokenizer.vocab),
            emb_dim, max_seq_len, num_heads, num_layers,
            dropout, device
        )
        self.emo_classifier = EmoClassifier(**emo_classifier_conf)
        self.emo_token_id = tokenizer.vocab.emo_token_id

        self.device = device

    def get_input_embeddings(self):
        return self.token_embs

    def set_input_embeddings(self, embeddings):
        self.token_embs = embeddings

    def forward(self, q_ids, q_lens, input_ids, attention_ids=None, emo_labels=None):
        """
        Args:
            q_ids: (N, seq_len)
            q_lens: (N)
            input_ids: (N, seq_len)
            attention_ids: (N, seq_len)
            emo_labels: (N) => used when appling teacher forcing

        Return:
            logits: (N, seq_len, vocab_size)
            Each of the logits indicate a confidence of each word for each position
            
        Example:
        <s> <emo> 기쁨 <sp1> '▁날씨가', '▁굉', '장', '히', '▁화', '창', '하', '네', '?' <sp2> '▁정말', '▁그렇', '네', '요.' </s>
        """
        
        # Emotion classifier (LSTM)
        q_input_embs = self.gpt.dropout(self.gpt.token_embs(q_ids))
        emo_logits = self.emo_classifier(q_input_embs, q_lens)
        emo_pred = torch.argmax(emo_logits, dim=-1) # (N)
        
        emo_prompts = torch.tensor(
            [Prompts[x] for x in (emo_pred.cpu().tolist() if emo_labels is None 
                            else emo_labels.cpu().tolist())], device=self.device
        )

        # GPT
        batch_size = input_ids.size(0)
        input_ids = torch.cat([
            input_ids[:, :1],
            torch.tensor(self.emo_token_id).repeat(batch_size, 1),
            emo_prompts.view(batch_size, 1),
            input_ids[:, 1:]
        ], dim=1)

        if attention_ids is not None:
            attention_ids = torch.cat([
                attention_ids[:, :1], 
                torch.ones(batch_size, 2),
                attention_ids[:, 1:]
            ], dim=1)

        lm_logits = self.gpt(input_ids, attention_ids)

        return emo_logits, lm_logits
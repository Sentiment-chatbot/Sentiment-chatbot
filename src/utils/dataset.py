from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import get_num_workers


class DialogueDataset(Dataset):
    def __init__(self, X, E, tokenizer, max_length=40):
        super().__init__()
        self.X = tokenizer(X.iloc[:, 0].to_list(), max_length=max_length)
        self.y = tokenizer(X.iloc[:, 1].to_list(), max_length=max_length)
        self.E = E

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        e = torch.tensor(self.E[idx])
        return x, y, e

    def __len__(self):
        return len(self.X)


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word("<PAD>")
        self.add_word("<SOS>")
        self.add_word("<EOS>")
        self.add_word("<UNK>")

        self.pad_token_id = self.word2idx["<PAD>"]
        self.sos_token_id = self.word2idx["<SOS>"]
        self.eos_token_id = self.word2idx["<EOS>"]
        self.unk_token_id = self.word2idx["<UNK>"]

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def make_vocab(self, sentence_list, threshold=5):
        counter = Counter()
        for i, sentence in enumerate(sentence_list):
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        for word, cnt in counter.items():
            if cnt >= threshold:
                self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<UNK>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(sentence):
        tokens = mecab.morphs(sentence)
        return tokens


class Collate(object):
  def __init__(self, pad_token_id):
    self.pad_token_id = pad_token_id
  
  def __call__(self, data):
    print(data)
    x, y, e = data
    x = pad_sequence(x, batch_first=True, padding_value=self.pad_token_id).long()
    y = pad_sequence(y, batch_first=True, padding_value=self.pad_token_id).long()
    e = torch.stack(e, dim=0)

    return x, y, e


def get_dataloader(dataset, pad_token_id, train=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=get_num_workers(),
        shuffle=train,
        collate_fn=Collate(pad_token_id=vocab.pad_token_id)
    )

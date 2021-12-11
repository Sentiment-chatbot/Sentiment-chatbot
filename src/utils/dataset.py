from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from soynlp.tokenizer import LTokenizer

from .utils import get_num_workers
from .tokenizer import Tokenizer


class Vocabulary(object):
    def __init__(self, base_tokenizer):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        if base_tokenizer == "Ltokenizer":
            self.base_tokenizer = LTokenizer()
        else:
            raise NotImplementedError

        self.add_word("<s>")
        self.add_word("</s>")
        self.add_word("<pad>")
        self.add_word("<unk>")
        self.add_word("<sp1>")
        self.add_word("<sp2>")
        self.add_word("<emo>")

        self.bos_token_id = self.word2idx["<s>"]
        self.eos_token_id = self.word2idx["</s>"]
        self.pad_token_id = self.word2idx["<pad>"]
        self.unk_token_id = self.word2idx["<unk>"]
        self.sp1_token_id = self.word2idx["<sp1>"]
        self.sp2_token_id = self.word2idx["<sp2>"]
        self.emo_token_id = self.word2idx["<emo>"]

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_sequence(self, sentence_list, threshold=3):
        counter = Counter()
        for i, sentence in enumerate(sentence_list):
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        for word, cnt in counter.items():
            if cnt >= threshold:
                self.add_word(word)

    def add_df(self, df):
        self.add_sequence(df['q'].tolist())
        self.add_sequence(df['a'].tolist())

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, sentence):
        try:
            tokens = self.base_tokenizer.tokenize(sentence)
        except:
            print(f"Fail to tokenize sentence: {sentence}")
            raise ValueError

        return tokens


class DialogueDataset(Dataset):
    """ Dialogue dataset for train/valid """
    def __init__(self, df, vocab, tokenizer):
        super().__init__()
        emotion_1_types = sorted(df.emotion_1.unique())
        emotion_1_dict = {name: i for i, name in enumerate(emotion_1_types)}
        
        emotion_2_types = sorted(df.emotion_2.unique())
        emotion_2_dict = {name: i for i, name in enumerate(emotion_2_types)}

        df = df.replace({
            'emotion_1': emotion_1_dict,
            'emotion_2': emotion_2_dict
        })

        self.input_ids = []
        self.emotion_1 = df.emotion_1.tolist()
        self.emotion_2 = df.emotion_2.tolist()

        q_set = tokenizer.encode(df.q.tolist())
        a_set = tokenizer.encode(df.a.tolist())
        for q, a in zip(q_set, a_set):
            ids = [vocab.bos_token_id, vocab.sp1_token_id] + q \
                    + [vocab.sp2_token_id] + a + [vocab.eos_token_id]
            self.input_ids.append(ids)

        self.q_ids = q_set
        self.q_lens = [len(q) for q in q_set]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.q_ids[idx]),
            torch.tensor(self.q_lens[idx]),
            torch.tensor(self.input_ids[idx]),
            torch.tensor(self.emotion_1[idx]),
            torch.tensor(self.emotion_2[idx])
        )
        
    def __len__(self):
        return len(self.input_ids)


class DialogueTestDataset(Dataset):
    """ Dialogue dataset for test """
    def __init__(self, df, vocab, tokenizer):
        super().__init__()
        emotion_1_types = sorted(df.emotion_1.unique())
        emotion_1_dict = {name: i for i, name in enumerate(emotion_1_types)}

        df = df.replace({
            'emotion_1': emotion_1_dict
        })

        self.input_ids = []
        self.input_raws = df.q.tolist()
        self.label_ids = []
        self.label_raws = df.a.tolist()
        self.emotion_1 = df.emotion_1.tolist()

        q_set = tokenizer.encode(self.input_raws)
        a_set = tokenizer.encode(self.label_raws)
        for q, a in zip(q_set, a_set):
            input_id = [vocab.bos_token_id, vocab.sp1_token_id] + q + [vocab.sp2_token_id]
            self.input_ids.append(input_id)
            self.label_ids.append(a + [vocab.eos_token_id])

        self.q_ids = q_set
        self.q_lens = [len(q) for q in q_set]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.q_ids[idx]),
            torch.tensor(self.q_lens[idx]),
            torch.tensor(self.input_ids[idx]),
            self.input_raws[idx],
            torch.tensor(self.label_ids[idx]),
            self.label_raws[idx],
            torch.tensor(self.emotion_1[idx]),
        )

    def __len__(self):
        return len(self.input_ids)


def load_data_loader(ds, pad_token_id, batch_size, shuffle=False):
    """ Data loader for train/validation """

    class Collate(object):
        def __init__(self, pad_token_id):
            self.pad_token_id = pad_token_id

        def __call__(self, data):
            q_ids, q_lens, input_ids, emotion_1, emotion_2 = zip(*data)
            input_ids = pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id
            ).long()
            q_ids = pad_sequence(
                q_ids,
                batch_first=True,
                padding_value=self.pad_token_id
            ).long()
            attention_ids = (input_ids != pad_token_id).float()
            q_lens = torch.stack(q_lens, dim=0)
            emotion_1 = torch.stack(emotion_1, dim=0)
            emotion_2 = torch.stack(emotion_2, dim=0)

            return q_ids, q_lens, input_ids, attention_ids, emotion_1, emotion_2

    num_workers = get_num_workers()
    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collate(pad_token_id=pad_token_id)
    )

    return data_loader

def load_test_loader(test_ds):
    """ Data loader for test (without batching) """

    num_workers = get_num_workers()
    test_data_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    return test_data_loader

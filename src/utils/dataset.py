from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from utils import get_num_workers
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from .tokenizer import Tokenizer
import os.path
import os
import pandas as pd


def get_path(path):
    train_path = os.path.join(path, "/train.csv")
    valid_path = os.path.join(path, "/valid.csv")
    return train_path, valid_path


def load_dataset(path):
    # path = os.path.dirname(os.getcwd())
    train_path, valid_path = get_path(path)
    train_df = pd.read_excel(train_path).fillna("") #df['사람문장1'].to_list()
    valid_df = pd.read_excel(valid_path).fillna("")

    emotion_1 = train_df["emotion_1"]
    train_df, test_df = train_test_split(train_df, test_size = 0.1, stratify=emotion_1)
    return train_df, valid_df, test_df


def tokenize_data(df, vocab):
    sentiment_dict = {"기쁨": 0, "당황":1, "상처":2, "슬픔":3, "불안":4, "분노":5}
    input_data = []
    output_data = []    
    sentiment = []
    tokenizer = Tokenizer()

    
    for i, row in df.iterrows():
        q1 = row["q1"]
        q2 = row["q2"]
        q3 = row["q3"]
        q4 = row["q4"]
        a1 = row["a1"]
        a2 = row["a2"]
        a3 = row["a3"]
        a4 = row["a4"]
        label = row["emotion_1"]

        if(q1 != ""):
            q1 = tokenizer.encode(q1)
            a1 = tokenizer.encode(a1)
            q1_ids = [vocab[vocab.bos_token_id]] + q1 + [vocab[vocab.eos_token_id]]
            a1_ids = [vocab[vocab.bos_token_id]] + a1 + [vocab[vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q1_ids))
            output_data.append(torch.tensor(a1_ids))
            sentiment.append(sentiment_dict[label])
        
        if(q2 != ""):
            q2 = tokenizer.encode(q2)
            a2 = tokenizer.encode(a2)
            q2_ids = [vocab[vocab.bos_token_id]] + q2 + [vocab[vocab.eos_token_id]]
            a2_ids = [vocab[vocab.bos_token_id]] + a2 + [vocab[vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q2_ids))
            output_data.append(torch.tensor(a2_ids))
            sentiment.append(sentiment_dict[label])

        if(q3 != ""):   
            q3 = tokenizer.encode(q3)
            a3 = tokenizer.encode(a3)
            q3_ids = [vocab[vocab.bos_token_id]] + q3 + [vocab[vocab.eos_token_id]]
            a3_ids = [vocab[vocab.bos_token_id]] + a3 + [vocab[vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q3_ids))
            output_data.append(torch.tensor(a3_ids))
            sentiment.append(sentiment_dict[label])

        if(q4 != ""):
            q4 = tokenizer.encode(q4)
            a4 = tokenizer.encode(a4)
            q4_ids = [vocab[vocab.bos_token_id]] + q4 + [vocab[vocab.eos_token_id]]
            a4_ids = [vocab[vocab.bos_token_id]] + a4 + [vocab[vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q4_ids))
            output_data.append(torch.tensor(a4_ids))
            sentiment.append(sentiment_dict[label])

    
    input_data = pad_sequence(input_data, padding_value=vocab.pad_token_id)
    output_data = pad_sequence(output_data, padding_value=vocab.pad_token_id)
    
    sentiment = torch.tensor(sentiment)

    dataset = TensorDataset(input_data, output_data, sentiment)
    
    return dataset


def get_data_loaders(train_data, valid_data, test_data, batch_size, shuffle=False):
    train_loader = DataLoader(
        train_data,
        num_workers=get_num_workers(),
        batch_size=batch_size
    )

    val_loader = DataLoader(
        valid_data,
        num_workers=get_num_workers(),
        batch_size=batch_size
    )

    test_loader = DataLoader(
        test_data,
        num_workers=get_num_workers(),
        batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


class Vocabulary(object):
    def __init__(self):
        
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.mecab = Mecab()

        self.add_word("<pad>")
        self.add_word("<s>")
        self.add_word("</s>")
        self.add_word("<unk>")

        self.pad_token_id = self.word2idx["<pad>"]
        self.bos_token_id = self.word2idx["<s>"]
        self.eos_token_id = self.word2idx["</s>"]
        self.unk_token_id = self.word2idx["<unk>"]

        

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def make_vocab(self, sentence_list, threshold=3):
        counter = Counter()
        for i, sentence in enumerate(sentence_list):
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        for word, cnt in counter.items():
            if cnt >= threshold:
                self.add_word(word)

    def vocab_data(self, df):
        self.df = df
        HS01 = self.df['q1'].to_list()
        HS02 = self.df['q2'].to_list()
        HS03 = self.df['q3'].to_list()
        HS04 = self.df['q4'].to_list()
        SS01 = self.df['a1'].to_list()
        SS02 = self.df['a2'].to_list()
        SS03 = self.df['a3'].to_list()
        SS04 = self.df['a4'].to_list()
        corpus = [HS01, HS02, HS03, HS04, SS01, SS02, SS03, SS04]
        for column in corpus:
            self.make_vocab(column)

    def start_vocab(self, train_df, valid_df, test_df):
        self.train = train_df
        self.valid = valid_df
        self.test = test_df
        self.vocab_data(self.train)
        self.vocab_data(self.valid)
        self.vocab_data(self.test)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    # @staticmethod
    def tokenize(self, sentence):
        tokens = self.mecab.morphs(sentence)
        return tokens




from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from soynlp.tokenizer import LTokenizer

from .utils import get_num_workers
from .tokenizer import Tokenizer


def tokenize_data(df, vocab):
    emotion_1_dict = {"기쁨": 0, "당황":1, "상처":2, "슬픔":3, "불안":4, "분노":5}
    emotion_2_dict = {'분노':0, '툴툴대는':1, '좌절한':2, '짜증내는':3, '방어적인':4, '악의적인':5, '안달하는':6, '구역질 나는':7, '노여워하는':8, '성가신':9, '슬픔':10, '실망한':11, '비통한':12,
                        '후회되는':13, '우울한':14, '마비된':15, '염세적인':16, '눈물이 나는':17, '낙담한':18, '환멸을 느끼는':19, '불안':20, '두려운':21, '스트레스 받는':22, '취약한':23, '혼란스러운':24,
                        '회의적인':25, '걱정스러운':26, '조심스러운':27, '초조한':28, '상처':29, '질투하는':30, '배신당한':31, '고립된':32, '충격 받은':33, '가난한':34, '불우한':35, '희생된':36, '억울한':37,
                        '괴로워하는':38, '버려진':39, '당황':40, '고립된(당황한)':41, '남의 시선을 의식하는':42, '외로운':43, '열등감':44, '죄책감의':45, '부끄러운':46, '혐오스러운':47, '한심한':48, '혼란스러운(당황한)':49,
                        '기쁨':50, '감사하는':51, '신뢰하는':52, '편안한':53, '만족스러운':54, '흥분':55, '느긋':56, '안도':57, '신이 난':58, '자신하는':59, '당혹스러운':60}

    df = df.replace({'emotion_1', emotion_1_dict})
    df = df.replace({'emotion_2', emotion_2_dict})

    input_data = []
    output_data = []    
    emotion_1 = []
    emotion_2 = []
    tokenizer = Tokenizer()
    
    q = tokenizer.encode(df.q)
    a = tokenizer.encode(df.a)

    input_data.append(torch.tensor(q))
    output_data.append(torch.tensor(a))
    
    emotion_1 = torch.tensor(df.emotion_1)
    emotion_2 = torch.tensor(df.emotion_2)

    dataset = TensorDataset(input_data, output_data, emotion_1, emotion_2)
    
    return dataset


class Collate(object):
  def __init__(self, pad_token_id):
    self.pad_token_id = pad_token_id
  
  def __call__(self, data):
    print(data)
    x, y, e1, e2 = data
    x = pad_sequence(x, batch_first=True, padding_value=self.pad_token_id).long()
    y = pad_sequence(y, batch_first=True, padding_value=self.pad_token_id).long()
    e1 = torch.stack(e1, dim=0)
    e2 = torch.stack(e2, dim=0)

    return x, y, e1, e2


def get_data_loaders(train_data, valid_data, test_data, vocab, batch_size, shuffle=False):
    train_loader = DataLoader(
        train_data,
        num_workers=get_num_workers(),
        batch_size=batch_size,
        collate_fn=Collate(pad_token_id=vocab.pad_token_id)
    )

    val_loader = DataLoader(
        valid_data,
        num_workers=get_num_workers(),
        batch_size=batch_size,
        collate_fn=Collate(pad_token_id=vocab.pad_token_id)
    )

    test_loader = DataLoader(
        test_data,
        num_workers=get_num_workers(),
        batch_size=batch_size,
        collate_fn=Collate(pad_token_id=vocab.pad_token_id)
    )

    return train_loader, val_loader, test_loader


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.tokenizer = LTokenizer()

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

    def add_sequence(self, sentence_list, threshold=3):
        counter = Counter()
        for i, sentence in enumerate(sentence_list):
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        for word, cnt in counter.items():
            if cnt >= threshold:
                self.add_word(word)

    def make_vocab(self, df):
        self.add_sequence(df['q'].to_list())
        self.add_sequence(df['a'].to_list())

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, sentence):
        try:
            tokens = self.tokenizer.tokenize(sentence)
        except:
            print(sentence)
            assert 0
        return tokens

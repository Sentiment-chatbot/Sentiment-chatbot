from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import get_num_workers
from konlpy.tag import Mecab



# class DialogueDataset(Dataset):
#     def __init__(self, X, E, tokenizer, max_length=40):
#         super().__init__()
#         self.X = tokenizer(X.iloc[:, 0].to_list(), max_length=max_length)
#         self.y = tokenizer(X.iloc[:, 1].to_list(), max_length=max_length)
#         self.E = E

#     def __getitem__(self, idx):
#         x = torch.tensor(self.X[idx])
#         y = torch.tensor(self.y[idx])
#         e = torch.tensor(self.E[idx])
#         return x, y, e

#     def __len__(self):
#         return len(self.X)



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
        HS01 = self.df['사람문장1'].to_list()
        HS02 = self.df['사람문장2'].to_list()
        HS03 = self.df['사람문장3'].to_list()
        HS04 = self.df['사람문장4'].to_list()
        SS01 = self.df['시스템응답1'].to_list()
        SS02 = self.df['시스템응답2'].to_list()
        SS03 = self.df['시스템응답3'].to_list()
        SS04 = self.df['시스템응답4'].to_list()
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


# class Collate(object):
#   def __init__(self, pad_token_id):
#     self.pad_token_id = pad_token_id
  
#   def __call__(self, data):
#     print(data)
#     x, y, e = data
#     x = pad_sequence(x, batch_first=True, padding_value=self.pad_token_id).long()
#     y = pad_sequence(y, batch_first=True, padding_value=self.pad_token_id).long()
#     e = torch.stack(e, dim=0)

#     return x, y, e


# def get_dataloader(dataset, pad_token_id, batch_size=batch_size, train=True):
#     return DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         num_workers=get_num_workers(),
#         shuffle=train,
#         collate_fn=Collate(pad_token_id=vocab.pad_token_id)
#     )




import numpy as np
import torch
import json
import os.path
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import PreTrainedTokenizerFast
from torch.nn.utils.rnn import pad_sequence
import utils
from utils.tokenizer import Tokenizer
# import dataset

# Training.xlsx에 length 800자 넘는 케이스는 직접 공백 삭제하는 식으로 수정처리했습니다.





def get_path(root):
    train_path = os.path.join(root, "data/training/training.xlsx")
    valid_path = os.path.join(root, "data/validation/validation.xlsx")
    return train_path, valid_path


def load_dataset():
    root = os.path.dirname(os.getcwd())
    train_path, valid_path = get_path(root)
    train_df = pd.read_excel(train_path).fillna("")
    valid_df = pd.read_excel(valid_path).fillna("")
    train_df, test_df = train_test_split(train_df, test_size = 0.1)
    return train_df, valid_df, test_df



class SentimentCorpus(Dataset):
    def __init__(self, train_df, val_df, test_df, vocab):
        self.sentiment_dict = {"기쁨": 0, "기쁨 ": 0, "당황":1, "상처":2, "슬픔":3, "불안":4, "불안 ":4, "분노":5} # 실제로는 6개
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.vocab = vocab

        self.tokenizer = Tokenizer()

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.init_data()
    
    def init_data(self):
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)
        self.test_data = self.load_data(self.test_df)


    def load_data(self, df):
        MAX_LEN = 40
        
        input_data = []
        output_data = []
        
        sentiment = []

        
        HS01 = df['사람문장1'].to_list()
        HS02 = df['사람문장2'].to_list()
        HS03 = df['사람문장3'].to_list()
        HS04 = df['사람문장4'].to_list()
        SS01 = df['시스템응답1'].to_list()
        SS02 = df['시스템응답2'].to_list()
        SS03 = df['시스템응답3'].to_list()
        SS04 = df['시스템응답4'].to_list()
        emotion_large = df['감정_대분류'].to_list()

        
        for (q1, a1, q2, a2, q3, a3, q4, a4, label) in zip(HS01, SS01, HS02, SS02, HS03, SS03, HS04, SS04, emotion_large):
            if(q1 == ""):
                break
            
            q1 = self.tokenizer.encode(q1)
            a1 = self.tokenizer.encode(a1)
            q1_ids = [self.vocab[self.vocab.bos_token_id]] + q1 + [self.vocab[self.vocab.eos_token_id]]
            a1_ids = [self.vocab[self.vocab.bos_token_id]] + a1 + [self.vocab[self.vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q1_ids))
            output_data.append(torch.tensor(a1_ids))
            sentiment.append(self.sentiment_dict[label])
            
            if(q2 == ""):
                break
            
            q2 = self.tokenizer.encode(q2)
            a2 = self.tokenizer.encode(a2)
            q2_ids = [self.vocab[self.vocab.bos_token_id]] + q2 + [self.vocab[self.vocab.eos_token_id]]
            a2_ids = [self.vocab[self.vocab.bos_token_id]] + a2 + [self.vocab[self.vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q2_ids))
            output_data.append(torch.tensor(a2_ids))
            sentiment.append(self.sentiment_dict[label])

            if(q3 == ""):
                break
            
            q3 = self.tokenizer.encode(q3)
            a3 = self.tokenizer.encode(a3)
            q3_ids = [self.vocab[self.vocab.bos_token_id]] + q3 + [self.vocab[self.vocab.eos_token_id]]
            a3_ids = [self.vocab[self.vocab.bos_token_id]] + a3 + [self.vocab[self.vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q3_ids))
            output_data.append(torch.tensor(a3_ids))
            sentiment.append(self.sentiment_dict[label])

            if(q4 == ""):
                break
            
            q4 = self.tokenizer.encode(q4)
            a4 = self.tokenizer.encode(a4)
            q4_ids = [self.vocab[self.vocab.bos_token_id]] + q4 + [self.vocab[self.vocab.eos_token_id]]
            a4_ids = [self.vocab[self.vocab.bos_token_id]] + a4 + [self.vocab[self.vocab.eos_token_id]]
            
            input_data.append(torch.tensor(q4_ids))
            output_data.append(torch.tensor(a4_ids))
            sentiment.append(self.sentiment_dict[label])

        
        input_data = pad_sequence(input_data, padding_value=self.vocab.pad_token_id)
        output_data = pad_sequence(output_data, padding_value=self.vocab.pad_token_id)
        
        sentiment = torch.tensor(sentiment)

        dataset = TensorDataset(input_data, output_data, sentiment)
        
        return dataset


    def get_data_loaders(self, batch_size=32, shuffle=False):
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size
        )

        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size
        )

        return train_loader, val_loader, test_loader
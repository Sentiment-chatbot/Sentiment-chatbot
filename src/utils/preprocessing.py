import os
import os.path as p
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import Vocabulary


def load_emotion_df(xls_path):
    """
    Funct:
        - Load dataframe of emotion dataset
        - Simple pre-processing
    """

    df = pd.read_excel(xls_path)

    df = df.rename(columns={
        "감정_대분류": "emotion_1",
        "감정_소분류": "emotion_2",
        "사람문장1": "q1",
        "시스템응답1": "a1",
        "사람문장2": "q2",
        "시스템응답2": "a2",
        "사람문장3": "q3",
        "시스템응답3": "a3",
        "사람문장4": "q4",
        "시스템응답4": "a4"
    })
    df = df.drop(["번호", "연령", "성별", "상황키워드", "신체질환"], axis=1)

    df["emotion_1"] = df["emotion_1"].str.strip()
    df["emotion_2"] = df["emotion_2"].str.strip()
    y = df["emotion_2"]

    return df, y

def split_single_turn(df):
    splited_df = pd.DataFrame(columns=['q', 'a', 'emotion_1', 'emotion_2'])

    pairs = [("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")]
    for pair in pairs:
        q, a = pair
        pair_df = pd.DataFrame({
            'q': df[q],
            'a': df[a],
            'emotion_1': df["emotion_1"],
            'emotion_2': df["emotion_2"]
        })
        splited_df = splited_df.append(pair_df, ignore_index=True)
    splited_df = splited_df.drop_duplicates(subset=None, keep='first', ignore_index=False)
    splited_df = splited_df.dropna(axis=0)

    return splited_df

def make_dataframe(src_path, dst_path):
    train_path = p.join(src_path, "raw/train.xlsx")
    valid_path = p.join(src_path, "raw/valid.xlsx")

    train_df, train_y = load_emotion_df(train_path)
    valid_df, _ = load_emotion_df(valid_path)

    train_df, test_df, *_ = train_test_split(train_df, train_y, test_size=0.1, stratify=train_y)

    for df, name in zip((train_df, valid_df, test_df), ("train", "valid", "test")):
        splited_df = split_single_turn(df)
        splited_df.to_csv(p.join(dst_path, f"{name}.csv"), sep='\t', na_rep="")

def get_dataframe(src_path):
    dfs = []
    for name in ("train", "valid", "test"):
        df = pd.read_csv(p.join(src_path, f"{name}.csv"), sep='\t')
        df = df.dropna(axis=0) # 추후 수정 필요, 왜 결측값 제거가 preprocessing에서 안되는지 모르겠음
        dfs.append(df)

    return dfs

def make_vocab(base_tokenizer, src_dfs, dst_path):
    vocabulary = Vocabulary(base_tokenizer)

    for df in src_dfs:
        vocabulary.add_df(df)

    with open(p.join(dst_path, f"vocab_{base_tokenizer}.pkl"), "wb") as f:
        pickle.dump(vocabulary, f)

def get_vocab(base_tokenizer, src_path):
    vocab = None
    with open(p.join(src_path, f"vocab_{base_tokenizer}.pkl"), "rb") as f:
        vocab = pickle.load(f)

    return vocab

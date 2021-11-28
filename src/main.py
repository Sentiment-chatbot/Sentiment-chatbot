import os
import os.path as p

import torch

from .model import GPT2Model
from .option import GPT2DefaultConfig, get_arg_parser
from .train import train
from .utils import set_seed
from .utils.tokenizer import Tokenizer
from .utils.preprocessing import (
    make_dataframe,
    get_dataframe,
    make_vocab,
    get_vocab
)
from .utils.dataset import DialogueDataset, load_data_loader
from .utils.generate import generate


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Data preprocessing (if needed)
    data_root = p.join(args.data_path, "processed/")
    os.makedirs(data_root, exist_ok=True)
    if not (
        p.exists(p.join(data_root, "train.csv"))
        and p.exists(p.join(data_root, "valid.csv"))
        and p.exists(p.join(data_root, "test.csv"))
    ):
        print("Proprocessing...")
        make_dataframe(src_path=args.data_path, dst_path=data_root)
        print("Finish.\n")
    else:
        print("Processed set already exists.")

    # Loading dataframe
    print("Make dataframe...")
    train_df, valid_df, test_df = get_dataframe(data_root)
    print("Finish.\n")

    # Loading vocabulary
    print("Make vocabulary & tokenizer...")
    if not p.exists(p.join(data_root, f"vocab_{args.base_tokenizer}.pkl")):
        print("Vocab preprocessing...")
        make_vocab(
            args.base_tokenizer,
            src_dfs=(train_df, valid_df),
            dst_path=data_root
        )
        print("Finish.")
    vocab = get_vocab(args.base_tokenizer, data_root)
    tokenizer = Tokenizer(vocab, args.base_tokenizer)
    print(f"Finish. Vocabulary size: {len(vocab)}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    print(f"Running on {device}.")

    print("Load datasets...")
    train_ds = DialogueDataset(train_df, vocab, tokenizer)
    valid_ds = DialogueDataset(valid_df, vocab, tokenizer)
    test_ds = DialogueDataset(test_df, vocab, tokenizer)
    print("Finish. \n")
    
    print("Load dataloaders...")
    train_loader = load_data_loader(train_ds, vocab.pad_token_id, args.batch_size, shuffle=True)
    valid_loader = load_data_loader(valid_ds, vocab.pad_token_id, args.batch_size)
    test_loader = load_data_loader(test_ds, vocab.pad_token_id, args.batch_size)
    print("Finish. \n")

    print("Get model...")
    model = GPT2Model(**GPT2DefaultConfig, vocab_size=len(vocab), device=device)

    # print("Start train.")
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=args.n_epochs,
        device=device,
        logging_step=300
    )
    print("Finish. \n")
    
    print("Start generation.")
    response_sentence = generate(
        "나 요즘 너무 우울해.",
        max_seq_len=20,
        model=model,
        tokenizer=tokenizer,
        gen_policy=args.gen_policy,
        device=device
    )
    print("입력: 나 요즘 너무 우울해.")
    print(f"출력: {response_sentence}")
    print("Finish. \n")

    print("All finished.")


if __name__ == '__main__':
    main()

#### python -m src.main --seed 42 --batch_size 64 --epoch 1 --learning-rate 1e-4

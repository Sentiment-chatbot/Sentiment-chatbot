import os
import os.path as p
import argparse

import torch

from .model import EmoGPT2
from .option import GPT2DefaultConfig, EmoClassifierDefaultConfig, get_arg_parser
from .utils import set_seed
from .utils.tokenizer import Tokenizer
from .utils.preprocessing import (
    make_dataframe,
    get_dataframe,
    make_vocab,
    get_vocab
)
from .utils.generate import generate_with_user_input


def main():
    parser = get_arg_parser()
    parser.add_argument('--weight-path', '-w', type=str, help='Path of model state weight')
    args = parser.parse_args()
    print(args)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"-- Running on {device}. -- ")

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

    # Loading model
    print("Get model...")
    model = EmoGPT2(
        **GPT2DefaultConfig,
        emo_classifier_conf=EmoClassifierDefaultConfig,
        tokenizer=tokenizer,
        device=device
    )
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.weight_path)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.weight_path, map_location=torch.device('cpu'))['model_state_dict'])
    print(f"Success to load the checkpoint: {args.weight_path}")

    # Start testing
    print("Start test.")
    user_input = input("Input(Want to get out? Please type 'STOP'): ")
    while user_input != "STOP":
        print(f"Generation ongoing...")
        print(f"Input: {user_input}")
        response_sentence = generate_with_user_input(
            user_input=user_input,
            max_seq_len=30,
            model=model,
            tokenizer=tokenizer,
            gen_policy=args.gen_policy,
            device=device
        )
        print(f"Output: {response_sentence}")
        print(f"---------------------------")
        user_input = input("Input(Want to get out? Please type 'STOP'): ")

    # Success
    print("All finished.")

if __name__ == '__main__':
    main()

# python -m src.main --DEBUG --logging-step 5 --batch-size 2 --gen-policy greedy
# python -m src.main --seed 42 --batch_size 64 --epoch 1 --learning-rate 1e-4

import os
import os.path as p
from datetime import datetime

import torch
import wandb

from .model import EmoGPT2
from .option import GPT2DefaultConfig, EmoClassifierDefaultConfig, get_arg_parser
from .train import train, test
from .utils import set_seed
from .utils.tokenizer import Tokenizer
from .utils.preprocessing import (
    make_dataframe,
    get_dataframe,
    make_vocab,
    get_vocab
)
from .utils.dataset import (
    DialogueDataset, load_data_loader,
    DialogueTestDataset, load_test_loader
)


def main():
    parser = get_arg_parser()
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
        print("Preprocessing...")
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

    # Loading dataset
    print("Load datasets...")
    train_ds = DialogueDataset(train_df, vocab, tokenizer)
    valid_ds = DialogueDataset(valid_df, vocab, tokenizer)
    test_ds = DialogueTestDataset(test_df, vocab, tokenizer)
    print("Finish. \n")

    # Loading dataloader
    print("Load dataloaders...")
    train_loader = load_data_loader(train_ds, vocab.pad_token_id, args.batch_size, shuffle=True)
    valid_loader = load_data_loader(valid_ds, vocab.pad_token_id, args.batch_size, shuffle=False)
    test_loader = load_test_loader(test_ds)
    print("Finish. \n")

    # Loading model
    print("Get model...")
    model = EmoGPT2(
        **GPT2DefaultConfig,
        emo_classifier_conf=EmoClassifierDefaultConfig,
        tokenizer=tokenizer,
        device=device
    )

    # Wandb link
    print("\n--Wandb initialization--")
    if args.debug:
        print("DEBUGGING MODE - Start without wandb")
        wandb.init(mode="disabled")
    else:
        print("Start train & test with wandb")
        wandb.init(
            project="Final project",
            entity="skku-2021-2-ap-team15",
        )
        wandb.config.update(args)
        wandb.run.name = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Start training
    print("Start train.")
    train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=args.n_epochs,
        gen_max_seq_len=20,
        gen_policy=args.gen_policy,
        gen_ex_input=args.gen_ex_input,
        device=device,
        learning_rate=args.learning_rate,
        logging_step=args.logging_step,
    )
    print("Finish.\n")

    # Start testing
    print("Start test.")
    rouges, bleus, perplexity = test(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        gen_policy=args.gen_policy,
        device=device
    )

    # Print metric scores
    for i, (rouge, bleu) in enumerate(zip(rouges, bleus)):
        print(f"BLEU-{i + 1}: {bleu: .4f} | ROUGE-{i + 1}: {rouge: .4f}")
    print(f"Perplexity: {perplexity: .4f}")
    print("Finish.\n")

    # Success
    print("All finished.")

if __name__ == '__main__':
    main()

# python -m src.main --DEBUG --logging-step 5 --batch-size 2 --gen-policy greedy
# python -m src.main --seed 42 --batch_size 64 --epoch 1 --learning-rate 1e-4

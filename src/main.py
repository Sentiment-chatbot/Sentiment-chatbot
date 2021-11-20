import os
import os.path as p

from .model import GPT2Model
from .utils import set_seed
from .utils.preprocessing import (
    make_dataframe,
    get_dataframe,
    make_vocab,
    get_vocab
)
from .option import get_arg_parser


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
    print("Make vocabulary...")
    if not p.exists(p.join(data_root, "/vocab.pkl")):
        print("Vocab preprocessing...")
        make_vocab(src_dfs=(valid_df, test_df), dst_path=data_root)
        print("Finish.")
    vocab = get_vocab(data_root)
    print("Finish.\n")

    # train_ds, valid_ds, test_ds = map(
    #     lambda df: load_dataset(df), (train_df, valid_df, test_df)
    # )

    # train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    # valid_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    # test_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)

    # model = GPT2Model()

    ### train, eval

    print("All finished.")


if __name__ == '__main__':
    main()

#### python3 main.py --seed 42 --batch_size 64 --epoch 1 --learning-rate 1e-4

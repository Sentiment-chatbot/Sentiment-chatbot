import argparse
import option
from utils import preprocessing, utils, dataset

def main():
    parser = option.get_option_arg_parser()
    args = parser.parse_args()

    utils.set_seed(args.seed)

    train_df, valid_df, test_df = dataset.load_dataset(path)
    vocab = dataset.Vocabulary()
    vocab.start_vocab(train_df, valid_df, test_df)

    train_tokenized = dataset.tokenize_data(train_df, vocab)
    valid_tokenized = dataset.tokenize_data(valid_df, vocab)
    test_tokenized = dataset.tokenize_data(test_df, vocab)
    train_loader, val_loader, test_loader = dataset.get_data_loaders(train_df, valid_df, test_df, batch_size=args.batchsize)

    print("finish")



if __name__ == '__main__':
    main()


#### python3 main.py --batchsize 64 --epoch 1 --learning-rate 1e-4 --seed 42
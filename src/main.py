import argparse
import option
from utils import preprocessing, utils, dataset

def main():
    SEED = 42
    utils.set_seed(SEED)

    parser = option.get_option_arg_parser()
    args = parser.parse_args()
    train_df, valid_df, test_df = preprocessing.load_dataset()
    vocab = dataset.Vocabulary()
    vocab.start_vocab(train_df, valid_df, test_df)

    SentimentCorpus = preprocessing.SentimentCorpus(train_df, valid_df, test_df, vocab)
    train_loader, val_loader, test_loader = SentimentCorpus.get_data_loaders(batch_size=args.batchsize)

    print("finish")



if __name__ == '__main__':
    main()


#### python3 main.py --batchsize 64 --epoch 1 --learning-rate 1e-4
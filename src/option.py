import argparse

GPT2DefaultConfig = {
    'vocab_size': 5000,
    'emb_dim': 768,
    'max_seq_len': 256,
    'num_heads': 12,
    'num_layers': 12,
    'dropout': 0.1
}

def get_option_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                        help='Learning rate')

    return parser




if __name__ == '__main__':
    parser = get_option_arg_parser()
    args = parser.parse_args()
    # print(json.dumps(args.__dict__, indent=2))

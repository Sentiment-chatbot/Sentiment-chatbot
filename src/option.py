import argparse

GPT2DefaultConfig = {
    'emb_dim': 768,
    'max_seq_len': 256,
    'num_heads': 12,
    'num_layers': 12,
    'dropout': 0.1
}

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--n-epochs', '-e', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--data-path', '-r', type=str, default="data/",
                        help='Data path')
    parser.add_argument('--base-tokenizer', '-bt', type=str, default="Ltokenizer",
                        help='Tokenizer type, option: Ltokenizer, Mecab')
    parser.add_argument('--gen-policy', '-gp', type=str, default="greedy",
                        help='Teneration policy, option: greedy, beam, topk')

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    # print(json.dumps(args.__dict__, indent=2))

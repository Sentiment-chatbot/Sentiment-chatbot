import argparse

import torch

GPT2DefaultConfig = {
    'emb_dim': 768,
    'max_seq_len': 256,
    'num_heads': 12,
    'num_layers': 12,
    'dropout': 0.1
}

EmoClassifierDefaultConfig = {
    'emb_dim': 768,
    'num_layers': 2,
    'hidden_dim' : 768,
    'num_classes' : 6,
    'dropout': 0.1,
    'bidirectional': True
}

Prompts = [9851, 13019, 3420, 3855, 5957, 22]
Prompt_word2idx = {
    '기뻐': 9851,
    '당황스러워': 13019,
    '불안해': 3420,
    '상처': 3855,
    '슬퍼': 5957,
    '화가': 22
}
Prompt_idx2word = {
    9851: '기뻐',
    13019: '당황스러워',
    3420: '불안해',
    3855: '상처',
    5957: '슬퍼',
    22: '화가' 
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
                        help='Teneration policy, option: greedy, top-p')
    parser.add_argument('--gen-ex-input', '-gei', type=str, default="나 요즘 너무 우울해.",
                        help='Sample input (ex. --gen-ex-input "나 요즘 너무 우울해.")')
    parser.add_argument('--logging-step', '-ls', type=int, default=150,
                        help='Logging step during train')
    parser.add_argument('--DEBUG', dest='debug', action='store_true',
                        help="Disable the wandb to log if debug option is true")
    parser.add_argument('--NO-DEBUG', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    # print(json.dumps(args.__dict__, indent=2))

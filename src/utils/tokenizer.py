from soynlp.tokenizer import LTokenizer


class Tokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.tokenizer = LTokenizer

    def __call__(self, x, max_length=None):
        tokens = None
        if isinstance(x, list): # mulitple sentences
            if max_length is None:
                max_length = 9999
                tokens = [self.encode(s)[:max_length] for s in x]
            else: # single sentence
                tokens = self.encode(x)
                if max_length is not None:
                    tokens = tokens[:max_length]

        return tokens

    def encode(self, sentence): # Encode words for only single sentence
        tokens = self.tokenizer.tokenize(sentence)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids): # Decode ids for only single sentence
        return self.convert_ids_to_sentence(ids)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token] if token in self.vocab.word2idx else self.vocab.unk_token_id)
        return ids

    def convert_ids_to_tokens(self, ids):
        return [self.vocab.idx2word[id] for id in ids]

    def convert_ids_to_sentence(self, ids):
        return ' '.join(self.vocab.convert_ids_to_tokens(ids))

from soynlp.tokenizer import LTokenizer


class Tokenizer(object):
    def __init__(self, vocab, base_tokenizer):
        self.vocab = vocab
        if base_tokenizer == "Ltokenizer":
            self.base_tokenizer = LTokenizer()
        else:
            raise NotImplementedError

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

    def encode(self, raw):
        tokens = None
        if isinstance(raw, list):
            token_list = []
            for sentence in raw:
                tokens = self.base_tokenizer.tokenize(sentence)
                token_list.append(self.convert_tokens_to_ids(tokens))
            return token_list
        elif isinstance(raw, str):
            tokens = self.base_tokenizer.tokenize(raw)
            return self.convert_tokens_to_ids(tokens)
        else:
            raise NotImplementedError

    def decode(self, ids): # Decode ids for only single sentence
        return self.convert_ids_to_sentence(ids)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(
                self.vocab.word2idx[token] if token in self.vocab.word2idx 
                                           else self.vocab.unk_token_id
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        return [self.vocab.idx2word[id] for id in ids]

    def convert_ids_to_sentence(self, ids):
        return ' '.join(self.convert_ids_to_tokens(ids))


class NgramTokenizer(object):
    def __init__(self ,n):
        self.tokenizer = LTokenizer
        self.n = n

    def n_gram_encode(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return self.convert_ngram_tokens(tokens)
    
    def convert_ngram_tokens(self, tokens):
        n_gram_tokens = []
        for i in len(tokens) - (self.n-1):
            n_token = None
            for j in range(self.n):
                n_token += tokens[i+j]
            n_gram_tokens.append(n_token)
        return n_gram_tokens

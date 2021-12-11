import numpy as np

from soynlp.tokenizer import LTokenizer
from nltk.translate.bleu_score import corpus_bleu

# Perplexity score
def perplexity_score(cross_entropy_val):
    """ 
    Args: 
        cross_entropy_val: float (not a torch.tensor)
    """
    return np.exp(cross_entropy_val)

# ROUGE-N
def rouge_n_score(refs, gens, n):
    recall = 0                  
    
    for r, g in zip(refs, gens):
        r = get_ngram(r, n)
        g = get_ngram(g, n)
        set_r = set(r)
        set_g = set(g)
        recall += (1 - (len(set_r - set_g) / len(r)))
    recall = recall / len(refs)

    return recall

# BLEU
def bleu_score(refs, gens, n):
    tokenizer = LTokenizer()
    refs = [[tokenizer.tokenize(ref)] for ref in refs]
    gens = [tokenizer.tokenize(gen) for gen in gens]

    weights = tuple((1 / n if i < n else 0.0) for i in range(4))
    bleu = corpus_bleu(refs, gens, weights=(1, 0, 0, 0))
    return bleu

def get_ngram(sentence, n):
    tokenizer = LTokenizer()
    tokens = tokenizer.tokenize(sentence)
    return convert_ngram_tokens(tokens, n)
    
def convert_ngram_tokens(tokens, n):
    n_gram_tokens = []
    if len(tokens) < n:
        n_token =""
        for i in range(len(tokens)):
            n_token += tokens[i]
        n_gram_tokens.append(n_token)

    else:
        for i in range(len(tokens) - (n - 1)):
            n_token = ""
            for j in range(n):
                n_token += tokens[i + j]
                n_gram_tokens.append(n_token)

    return n_gram_tokens
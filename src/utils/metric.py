import numpy as np

# from KoBERTScore import BERTScore
from soynlp.tokenizer import LTokenizer
import nltk.translate.bleu_score as bleu

# Perplexity score
def perplexity_score(cross_entropy_val):
    """ 
    Args: 
        cross_entropy_val: float (not a torch.tensor)
    """
    return np.exp(cross_entropy_val)

# BERT score
# def koBERTscore(ref, gen):
#     model_name = "beomi/kcbert-base"
#     bertscore = BERTScore(model_name, best_layer=4)
#     bertscore(ref, gen, batch_size=128)
#     # [0.5643115, 0.4720116, 0.2556618, 0.2268927]

# ROUGE-N
def rouge_n_score(ref, gen, n):
    recall = 0                  
    
    for r, g in zip(ref, gen):
        r = get_ngram(r, n)
        g = get_ngram(g, n)
        set_r = set(r)
        set_g = set(g)
        recall += (1 - (len(set_r - set_g) / len(r)))
    recall = recall / len(ref)

    return recall

# BLEU
def bleu_score(refs, gen):
    return bleu.sentence_bleu(list(map(lambda ref: ref.split(), refs)),gen.split())

def get_ngram(sentence, n):
    tokenizer = LTokenizer()
    tokens = tokenizer.tokenize(sentence)
    return convert_ngram_tokens(tokens, n)
    
def convert_ngram_tokens(tokens, n):
    n_gram_tokens = []
    for i in range(len(tokens) - (n-1)):
        n_token = ""
        for j in range(n):
            n_token += tokens[i+j]
        n_gram_tokens.append(n_token)
    return n_gram_tokens


# references = [
#     '날씨는 좋고 하지만 할일은 많고 끝났다 어우'#,
#     # '이 영화 정말 재밌었어요',
#     # '점수가 낮은 문장',
#     ]
# candidates = [
#     '날씨는 좋고 하지만 할일은 많고 어우'#,
#     # '영화 정말 재밌었어요 잘 고른거 같아',
#     # '브라질 열대우림이 장기간 화재로 면적이 줄어들고 있습니다',
# ]

# candidate = '날씨는 좋고 하지만 할일은 많고 어우'


# print("handmade BLEU : {}".format(BLEU(references, candidates, 4)))
# print("nltk BLEU : {}".format(bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),candidate.split())))

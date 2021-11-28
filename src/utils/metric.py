from KoBERTScore import BERTScore
from soynlp.tokenizer import LTokenizer
import nltk.translate.bleu_score as bleu


def koBERTscore(ref, gen):
    model_name = "beomi/kcbert-base"
    bertscore = BERTScore(model_name, best_layer=4)
    bertscore(ref, gen, batch_size=128)
    # [0.5643115, 0.4720116, 0.2556618, 0.2268927]

def ROUGE_n(ref, gen, n):
    recall = 0                  
    
    for r, g in zip(ref, gen):
        r = get_ngram(r, n)
        g = get_ngram(g, n)
        Set_r = set(r)
        Set_g = set(g)
        recall += 1 - (len(Set_r - Set_g) / len(r))
        # print("recall : {}".format(precision, recall))
    recall = recall/len(ref)

    return recall

def BLEU(ref, gen, n):
    precision = 0               # BLEU
    
    for r, g in zip(ref, gen):
        r = get_ngram(r, n)
        g = get_ngram(g, n)
        Set_r = set(r)
        Set_g = set(g)
        precision += 1 - (len(Set_g - Set_r) / len(g))
        # print("precision : {}\nrecall : {}".format(precision, recall))
    precision = precision/len(ref)

    return precision

def nltk_BLEU(refs, gen):
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

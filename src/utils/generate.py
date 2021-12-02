import random

import numpy as np
import torch
import torch.nn.functional as F

from .metric import perplexity_score, rouge_n_score


def log(number):
    # log에 0이 들어가는 것을 막기 위해 아주 작은 수를 더해줌.
    return np.log(number + 1e-10)

def greedy_search(prediction):
    # prediction(torch.tensor): seq_len, vocab_size
    return torch.argmax(prediction[-1, :])

def beam_search(predictions, k=None):
    # prediction: (seq_len, vocab_size)
    sequences = [[list(), 1.0]]

    for row in predictions:
        all_candidates = list()

        # 1. 각각의 timestep에서 가능한 후보군으로 확장
        for i in range(len(sequences)):
            seq, score = sequences[i]

            # 2. 확장된 후보 스텝에 대해 점수 계산
            for j in range(len(row)):
                new_seq = seq + [j]
                new_score = score * -log(row[j])
                candidate = [new_seq, new_score]
                all_candidates.append(candidate)

        # 3. 가능도가 높은 k개의 시퀀스만 남김
        ordered = sorted(all_candidates, key=lambda tup: tup[1])  # 점수 기준 정렬
        sequences = ordered[:k]

    return sequences

def generate_with_user_input(
    user_input,
    max_seq_len,
    model,
    tokenizer,
    gen_policy,
    device
):
    """ Inference (generation) 
    Args:
        user_input(str): User input 
                ex) "나 요즘 너무 우울해."
        max_seq_len(int): Maximum length of generated sentence
        model, tokenizer: Model, tokenizer
        gen_policy: Generation policy 
                i.e) Greedy search, Beam search, top-k search, etc.
        device: Device
    Return:
        pred_sentence(str): Generated output
                ex) "힘드시겠어요. 힘내세요." (expected)
    """ 
    generate_fn = None
    if gen_policy == "greedy":
        generate_fn = greedy_search
    else:
        raise NotImplementedError
    
    model.to(device)
    model.eval()
    
    input_ids = torch.tensor(
        [tokenizer.vocab.bos_token_id, tokenizer.vocab.sp1_token_id] + \
            tokenizer.encode(user_input) + [tokenizer.vocab.sp2_token_id],
        device=device
    ).unsqueeze(0) # (1, seq_len(sp1))

    # input_ids: <s> <sp1> {sentence1} <sp2>
    # pred: {sentence2} </s>
    pred_ids = []
    with torch.no_grad():
        for _ in range(max_seq_len):
            logits = model(input_ids).squeeze(0) # (seq_len, vocab_size)
            pred = generate_fn(logits) # single index
            
            if(pred.item() == tokenizer.vocab.eos_token_id):
                break
            pred_ids.append(pred.item())

            input_ids = torch.cat((input_ids, pred.view(1, 1)), dim=-1)

    pred_sentence = tokenizer.decode(pred_ids)
    return pred_sentence

def generate_with_data_loader(
    model,
    test_loader,
    tokenizer,
    gen_policy,
    device
):
    """ Test (WITHOUT teacher forcing) 
    Args:
        model, tokenizer: Model, tokenizer
        test_loader(DataLoader): Test loader
        gen_policy: Generation policy 
                i.e) Greedy search, Beam search, top-k search, etc.
        device: Device
    Return:
        Score based on metrics (perplexity, ROUGE-N, etc.)
    """ 

    generate_fn = None
    if gen_policy == "greedy":
        generate_fn = greedy_search
    else:
        raise NotImplementedError
    
    model.to(device)
    model.eval()
    # input_ids: <s> <sp1> {sentence1} <sp2>
    # pred: {sentence2} </s>

    pred_sentences = []
    input_sentences = []
    label_sentences = []
    perplexities = []
    with torch.no_grad():
        epoch_loss = 0.0
        for step, (input_ids, input_raws, label_ids, label_raws) in enumerate(test_loader):
            input_ids, label_ids = input_ids.to(device), label_ids.to(device) # (1, q_len), (1, a_len)

            pred_ids = []
            pred_logits = []
            while True:
                logits = model(input_ids).squeeze(0) # (seq_len, vocab_size)
                pred_logits.append(logits[-1, :]) # (vocab_size)
                pred = generate_fn(logits) # single index
                pred_ids.append(pred.item())

                if len(pred_ids) == label_ids.size(-1):
                    break
                input_ids = torch.cat((input_ids, pred.view(1, 1)), dim=-1)

            pred_sentences.append(tokenizer.decode(pred_ids))
            input_sentences.append(input_raws[0])
            label_sentences.append(label_raws[0])

            cross_entropy = F.cross_entropy(
                input=torch.stack(pred_logits, dim=0), # (a_len, vocab_size)
                target=label_ids.squeeze(0) # (a_len)
            ).cpu().numpy()
            perplexities.append(perplexity_score(cross_entropy))

            if (step + 1) % 1000 == 0:
                print(f"[Step {step + 1}/{len(test_loader)}] Ongoing...")

    rouge = rouge_n_score(ref=label_sentences, gen=pred_sentences, n=3)
    perplexity = np.average(perplexities)

    print("--Example--")
    for i in range(5):
        idx = random.randint(0, len(pred_sentences) - 1)
        print(f"{i}-th example")
        print(f"Input: {input_sentences[idx]}")
        print(f"Output: {pred_sentences[idx]}")
        print(f"Label: {label_sentences[idx]}")    
        print("-------")

    return rouge, perplexity

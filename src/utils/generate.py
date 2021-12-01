import numpy as np
import torch


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

def generate(
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

import random

import numpy as np
import torch
import torch.nn.functional as F

from .metric import perplexity_score, rouge_n_score, bleu_score


def greedy_search(model, q_ids, q_lens, input_ids):
    """ Greedy search strategy for text generation
    Args:
        model: model
        input_ids: size(1, seq_len)
        predictions(torch.tensor): size(L, seq_len, vocab_size)
    Return:
        pred(torch.tensor, size(1)): Predicted word vocab index
        logits: logits
    """

    with torch.no_grad():
        emo_out, out = model(
            q_ids=q_ids,
            q_lens=q_lens,
            input_ids=input_ids
        ) # (1, seq_len, vocab_size)
        logits = out.squeeze(0)[-1, :] # (vocab_size)
        pred = torch.argmax(logits) # (1)
        emo_pred = torch.argmax(emo_out, dim=1).item()

    return emo_pred, pred, logits

# TODO Not fully implemented
# def top_p_sampling(model, input_ids, p=0.9):
#     """ Top-p (Nucleus) sampling strategy for text generation
#     Args:
#         model: model
#         input_ids: size(1, seq_len)
#         predictions(torch.tensor): size(L, seq_len, vocab_size)
#     Return:
#         pred(torch.tensor, size(1)): Predicted word vocab index
#         logits: logits
#     """

#     _, logits = greedy_search(model, input_ids)
#     sorted_logits, idxs = torch.sort(logits, descending=True)
#     probs = F.softmax(sorted_logits, dim=-1) # (vocab_size)

#     cut_idx = -1
#     cumsum = 0.0
#     for i, prob in enumerate(probs):
#         cumsum += prob.cpu().item()
#         if cumsum >= p:
#             cut_idx = i
#             break

#     logits[idxs[cut_idx + 1:]] = float('-inf')
#     pred = torch.multinomial(F.softmax(logits, dim=-1), 1)

#     return pred, logits

# TODO Not fully implemented
# def beam_search(predictions, k=5):
#     """ Beam search strategy for text generation

#     Args:
#         predictions(torch.tensor): seq_len, vocab_size
#     """
#     if k != 5:
#         raise NotImplementedError

#     epsilon = 1e-10
#     sequences = [[list(), 1.0]]

#     for row in predictions:
#         all_candidates = list()

#         for i in range(len(sequences)):
#             seq, score = sequences[i]

#             for j in range(len(row)):
#                 new_seq = seq + [j]
#                 new_score = score * -np.log(row[j] + epsilon)
#                 candidate = [new_seq, new_score]
#                 all_candidates.append(candidate)

#         ordered = sorted(all_candidates, key=lambda tup: tup[1])
#         sequences = ordered[:k]

#     return sequences

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
    
    q_ids = torch.tensor(tokenizer.encode(user_input), device=device).unsqueeze(0)
    q_lens = torch.tensor([q_ids.size(-1)])
    input_ids = torch.tensor(
        [tokenizer.vocab.bos_token_id, tokenizer.vocab.sp1_token_id] + \
            tokenizer.encode(user_input) + [tokenizer.vocab.sp2_token_id],
        device=device
    ).unsqueeze(0) # (1, seq_len(sp1))

    # input_ids: <s> <sp1> {sentence1} <sp2>
    # pred: {sentence2} </s>
    pred_ids = []
    for _ in range(max_seq_len):
        _, pred, logits = generate_fn(model, q_ids, q_lens, input_ids)
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
    epoch_loss = 0.0

    emo_correct = 0
    for step, (q_ids, q_lens, input_ids, input_raws, label_ids, label_raws, emo_label) in enumerate(test_loader):
        q_ids, q_lens = q_ids.to(device), q_lens.to(device)
        input_ids, label_ids = input_ids.to(device), label_ids.to(device) # (1, q_len), (1, a_len)

        pred_ids = []
        pred_logits = []
        while len(pred_ids) != label_ids.size(-1):
            emo_pred, pred, logits = generate_fn(model, q_ids, q_lens, input_ids) # single index
            pred_logits.append(logits)
            pred_ids.append(pred.item())
            input_ids = torch.cat((input_ids, pred.view(1, 1)), dim=-1)

        pred_sentences.append(tokenizer.decode(pred_ids))
        input_sentences.append(input_raws[0])
        label_sentences.append(label_raws[0])

        if emo_pred == emo_label:
            emo_correct += 1
        cross_entropy = F.cross_entropy(
            input=torch.stack(pred_logits, dim=0), # (a_len, vocab_size)
            target=label_ids.squeeze(0) # (a_len)
        ).cpu().numpy()
        perplexities.append(perplexity_score(cross_entropy))

        if (step + 1) % 1000 == 0:
            print(f"[Step {step + 1}/{len(test_loader)}] Ongoing...")

    rouges = [rouge_n_score(
        refs=label_sentences,
        gens=pred_sentences, n=n
    ) for n in range(1, 4)]

    bleus = [bleu_score(
        refs=label_sentences,
        gens=pred_sentences, n=n
    ) for n in range(1, 4)]

    perplexity = np.average(perplexities)

    print("--Example--")
    for i in range(5):
        idx = random.randint(0, len(pred_sentences) - 1)
        print(f"{i}-th example")
        print(f"Input: {input_sentences[idx]}")
        print(f"Output: {pred_sentences[idx]}")
        print(f"Label: {label_sentences[idx]}")    
        print("-------")
    print(f"Emotion ACC: {emo_correct / len(test_loader.dataset):.3f}")
    return rouges, bleus, perplexity

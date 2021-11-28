import numpy as np
import torch


def log(number):
    # log에 0이 들어가는 것을 막기 위해 아주 작은 수를 더해줌.
    return np.log(number + 1e-10)

def greedy_search(prediction, k, target_idx):
    # prediction: (seq_len, vocab_size)
    return torch.argmax(prediction[target_idx])

def beam_search(predictions, k):
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

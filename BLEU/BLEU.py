from collections import Counter
import numpy as np
from nltk import ngrams

class BLEU:
    def __init__(self):
        pass

    def simple_count(self, tokens, n):
        return Counter(ngrams(tokens, n))

    def count_clip(self, candidate, reference_list, n):
        cnt_ca = self.simple_count(candidate, n)
        temp = dict()

        for ref in reference_list:
            cnt_ref = self.simple_count(ref, n)
            for n_gram in cnt_ref:
                if n_gram in temp:
                    temp[n_gram] = max(cnt_ref[n_gram], temp[n_gram])  # 다른 ref들과 비교했을 때 더 많은 걸 선택
                else:
                    temp[n_gram] = cnt_ref[n_gram]

        return {n_gram: min(cnt_ca.get(n_gram, 0), temp.get(n_gram, 0)) for n_gram in cnt_ca}

    def modified_precision(self, candidate, reference_list, n):
        clip = self.count_clip(candidate, reference_list, n)
        total_clip = sum(clip.values())

        ct = self.simple_count(candidate, n)
        total_ct = sum(ct.values())
        if total_ct == 0:
            total_ct = 1
        return (total_clip / total_ct)

    def closest_ref_length(self, candidate, reference_list):
        ca_len = len(candidate)
        ref_lens = [len(ref) for ref in reference_list]
        # ref_lens = [16, 18, 16] 이고, ca_len이 18이면, closest_ref_len = 18이다.
        closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - ca_len))
        return closest_ref_len

    def brevity_penalty(self, candidate, reference_list):
        ca_len = len(candidate)
        ref_len = self.closest_ref_length(candidate, reference_list)

        if ca_len > ref_len:
            return 1
        elif ca_len == 0:
            return 0
        else:
            return np.exp(1 - ref_len / ca_len)

    def bleu_score(self, candidate, reference_list, weights=[0.25] * 4):
        bp = self.brevity_penalty(candidate, reference_list)
        p_n = [self.modified_precision(candidate, reference_list, n=n) for n, _ in enumerate(weights, start=1)]

        score = np.sum([w_i * np.log(p_i) if p_i != 0 else 0 for w_i, p_i in zip(weights, p_n)])
        return bp * np.exp(score)
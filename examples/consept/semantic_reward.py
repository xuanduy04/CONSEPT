import math
import re
from collections import Counter

import torch
import torch.nn.functional as F


def map_to_range(x, low, high):
    return torch.clamp((x - low) / (high - low), min=0.0, high=1.0)


_TOKEN_RE = re.compile(r"\b\w+\b")
def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def tfidf_cosine_similarity(s1: str, s2: str) -> float:
    docs = [tokenize(s1), tokenize(s2)]
    vocab = sorted(set(docs[0]) | set(docs[1]))
    N = len(docs)

    # IDF: log(N / doc_freq) for each term
    doc_freq = Counter(w for doc in docs for w in set(doc))
    idf = torch.tensor([math.log(N / (doc_freq[w])) for w in vocab])

    # TF: raw count / doc length
    def tf_vec(tokens):
        counts = Counter(tokens)
        vec = torch.tensor([counts.get(w, 0) / len(tokens) for w in vocab])
        return vec

    v1 = (tf_vec(docs[0]) * idf).unsqueeze(0)
    v2 = (tf_vec(docs[1]) * idf).unsqueeze(0)

    return F.cosine_similarity(v1, v2).item()


PRUNED_PROMPT_KEY = "[PRUNED]"
def get_semantic_reward(eos_token: str) -> callable:
    def semantic_reward(prompts: list[str], completions: list[str], solution: list[str], **kwargs) -> list[float]:
        r"""
        Reward function that checks if the completion is **semantically** the same as the ground truth.

        Args:
            prompts (`list[str]`):
                List of prompts that the model inputted
            completions (`list[str]`):
                List of completions to be evaluated.
            solution: (`list[str]`):
                List of the raw-text solutions to the questions/problems/prompts.
            **kwargs:
                Additional keyword arguments. This function does not use them, but they are required in the function
                signature to ensure compatibility with trainers like [`CONSEPTTrainer`].
        """
        contents = completions
        rewards = []
        sum_valid_rewards = 0.0
        valid_prompts = 0
        for prompt, content, _solution in zip(prompts, contents, solution):
            if prompt != eos_token:
                reward = tfidf_cosine_similarity(content, _solution)
                sum_valid_rewards += reward
                valid_prompts += 1
            else:
                reward = PRUNED_PROMPT_KEY
            rewards.append(reward)

        avg_rewards = float(sum_valid_rewards / valid_prompts) if valid_prompts else 0.
        return [reward if reward != PRUNED_PROMPT_KEY else avg_rewards for reward in rewards]
    return semantic_reward

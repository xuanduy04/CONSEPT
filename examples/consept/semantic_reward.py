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


def compute_tf(doc_tokens):
    counts = Counter(doc_tokens)
    total = len(doc_tokens)
    return {word: count / total for word, count in counts.items()}


def compute_idf(docs_tokens):
    N = len(docs_tokens)
    all_words = {word for doc in docs_tokens for word in doc}
    return {word: math.log((N + 1) / (sum(1 for doc in docs_tokens if word in doc) + 1)) + 1 for word in all_words}


def compute_tfidf_vector(tf, idf) -> "torch.Tensor":
    return torch.tensor([tf.get(word, 0.0) * idf[word] for word in idf])


def tfidf_cosine_similarity(s1: str, s2: str) -> float:
    docs = [tokenize(s1), tokenize(s2)]
    idf = compute_idf(docs)
    v1 = compute_tfidf_vector(compute_tf(docs[0]), idf).unsqueeze(0)
    v2 = compute_tfidf_vector(compute_tf(docs[1]), idf).unsqueeze(0)

    return F.cosine_similarity(v1, v2).item()


PRUNED_PROMPT_KEY = "<|PRUNED_PROMPT|>"


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

        avg_rewards = float(sum_valid_rewards / valid_prompts) if valid_prompts else 0.0
        return [reward if reward != PRUNED_PROMPT_KEY else avg_rewards for reward in rewards]

    return semantic_reward

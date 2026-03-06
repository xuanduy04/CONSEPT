def semantic_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
    r"""
    Reward function that checks if the completion is **semantically** the same as the ground truth.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`CONSEPTTrainer`].
    """
    contents = completions
    rewards = []
    for content, solution in zip(contents, solution):
        reward = 1.0  # TODO: implement
        rewards.append(reward)

    return rewards

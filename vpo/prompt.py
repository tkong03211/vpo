"""Multi-answer prompt formatting.

VPO asks the model for m answers in a SINGLE autoregressive chain, separated
by a delimiter, so each answer can attend to the previous ones — this is where
the in-context diversity steering comes from (Section 3.1). These helpers just
build that instruction and split the chain back into answers.
"""

DEFAULT_DELIMITER = "\n---\n"


def format_multi_answer_prompt(
    prompt: str, num_answers: int = 3, delimiter: str = DEFAULT_DELIMITER
) -> str:
    """Wrap a prompt with an instruction to emit m delimiter-separated answers.

    The single-chain format lets later answers condition on earlier ones.

    prompt: the user task
    num_answers: m — how many answers to request
    returns: the augmented prompt string
    """
    sep = delimiter.strip()
    return (
        f"{prompt}\n\n"
        f"Give {num_answers} different answers, each separated by a line "
        f"containing only '{sep}'. Make the answers as distinct as possible."
    )


def split_answers(completion: str, delimiter: str = DEFAULT_DELIMITER) -> list[str]:
    """Split a model completion back into its individual answers.

    Inverse of format_multi_answer_prompt's delimiter convention.

    completion: the raw model output containing delimiter-separated answers
    returns: list of stripped answer strings (empty pieces dropped)
    """
    sep = delimiter.strip()
    parts = [piece.strip() for piece in completion.split(sep)]
    return [piece for piece in parts if piece]

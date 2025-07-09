import re
from typing import Optional

from termcolor import colored


def parse_and_check_numerical_answer(
    answer, min=None, max=None, final_score=False, provide_explanations=False
):
    attempt = parse_numerical_answer(
        answer, min, max, final_score, provide_explanations
    )
    if attempt is not None:
        score = attempt[0] if provide_explanations else attempt
        if score < min or score > max:
            return None
        return attempt

    return None


def parse_final_score(
    answer: str, min_score: int = 0, max_score: int = 100
) -> Optional[int]:
    """
    Extracts the final integer score between min_score and max_score (inclusive) from a string, robust to Markdown,
    labels, and formats. Returns the score as an int, or None if not found/invalid.

    Args:
        answer: The string containing the score, which may include Markdown formatting, labels, etc.
        min_score: Minimum valid score. Default: 0.
        max_score: Maximum valid score. Default: 100.

    Returns:
        The extracted score if valid, otherwise None.
    """
    # Remove Markdown bold/italic formatting
    cleaned = re.sub(r"[*_]+", "", answer)

    # Look for all numbers in the string
    numbers = re.findall(r"\b\d{1,3}\b", cleaned)
    # If nothing found, give up
    if not numbers:
        return None

    # Convert all found numbers to ints
    numbers = [int(num) for num in numbers]
    # Filter to only valid score range
    valid_scores = [num for num in numbers if min_score <= num <= max_score]

    if not valid_scores:
        return None

    # Choose the LAST valid score (per TEMPLATE_GEMBA_ESA_RANKING_FOR_COMMAND_A prompt/instructions)
    return valid_scores[-1]


def parse_numerical_answer(
    answer, min=None, max=None, final_score=False, provide_explanations=False
):
    if final_score:
        score = parse_final_score(answer, min, max)
        if score is not None:
            return (score, answer) if provide_explanations else score

    # get all numbers in a string
    numbers = re.findall(r"\d+", answer)
    if len(numbers) == 1:
        return (int(numbers[0]), answer) if provide_explanations else int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return (
            (int(answer[2:-2]), answer) if provide_explanations else int(answer[2:-2])
        )

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return (
                (int(answer.split("/")[0]), answer)
                if provide_explanations
                else int(answer.split("/")[0])
            )

    return None


def validate_number(x, min=0, max=100, final_score=False, provide_explanations=False):
    attempt = parse_and_check_numerical_answer(
        x, min, max, final_score, provide_explanations
    )
    if attempt is not None:
        return attempt
    return None

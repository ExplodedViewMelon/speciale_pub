from models import *
from typing import Literal


def get_recall(answer: str, predictions: list[str]) -> int:
    """
    Returns index of answer in predictions
    """
    for i, prediction in enumerate(predictions):
        if prediction == answer:
            return i
    return -1


def print_predictions(
    answer: str, answer_position: int, predictions: list[str]
) -> None:
    if answer_position == -1:
        print(f"\033[91m{answer}\033[0m not in", predictions)
    else:
        print("      ", end="")
        print("[", end="")
        for n, prediction in enumerate(predictions):
            if n == answer_position:
                print(f"\033[92m{prediction}\033[0m", end=", ")
            else:
                print(prediction, end=", ")
        print("]", end="")
        print("")


def get_openai_token():
    with open("openai_api.txt", "r") as file:
        api_key = file.read().strip()
    return api_key


def get_multiple_predictions_by_elemination(
    model: Model, symptoms: str, options: set[str], top_k: int
) -> list[str]:
    """
    Itaratively get predictions from symptoms by eleminating previous answers from options
    """
    # Sample predictions by eliminating options
    predictions_raw = []
    options_copy = options.copy()
    for _ in range(top_k):
        # update answer class
        class AnswerOptionsShrinking(BaseModel):
            reasoning: str
            disease: Literal[tuple(options_copy)]  # type: ignore

        prediction_raw: Any = model.complete(
            f"Which disease from the options are causing the following symptoms?\n\nSymptoms: {symptoms}\n\nOptions: {options}",
            AnswerOptionsShrinking,
            max_retries=0,  # disable retrying
            model="gpt-4o",
        )
        predictions_raw.append(prediction_raw)
        if prediction_raw:
            try:
                options_copy.remove(prediction_raw.disease)
            except:
                print("prediction not in options - cannot remove")

    # Remove None if any
    predictions: list[str] = [
        prediction.disease for prediction in predictions_raw if prediction
    ]

    return predictions


def get_semantic_grounding_mulitple(
    model: Model, predictions: list[str], symptoms: str
) -> list[tuple[str, int, dict]]:
    """
    Returns list of tuples, for each prediction:
        prediction (str), total length of passages (int), {symptom (str):[supporting passages (str)]})
    Using knn on latent repr. of each symptom and each sentence from fz retrieved material queried on prediction.
    Each passage are validated using the llm and symptoms are filtered with criteria >= 2 supporting passages.
    """

    from semantic_grounding import grounder_retrieve_passages

    prediction_passages: list[tuple[str, int, dict]] = []
    for prediction in predictions:
        grounding, n_passages = grounder_retrieve_passages(model, prediction, symptoms)
        prediction_passages.append((prediction, n_passages, grounding))

        # for key, value in grounding.items():
        #     print(f"Key: {key}, n-passages: {len(value)}")
    return prediction_passages


def get_recall_free(model: Model, answer: str, predictions: list[str]) -> int:
    class AnswerIsSynomymous(BaseModel):
        is_synonymous: bool

    for prediction in predictions:
        prompt = (
            "You are to evaluate if the two following medical terms are somewhat synomymous. They do not need to be exactly equal, just the same root disease."
            f"Is {prediction} synonymous with {answer}?"
        )
        model.complete_add_to_batch_queue(prompt, model="gpt-4o")

    predictions_are_correct: list[bool] = [
        answer.is_synonymous
        for answer in model.complete_execute_queue(AnswerIsSynomymous, model="gpt-4o")
    ]

    # Get recall-at-k metric
    answer_position = -1
    for i, are_correct in enumerate(predictions_are_correct):
        if are_correct:
            return i
    return -1


class AnswerFree(BaseModel):
    reasoning: str
    disease: str


class AnswerIsDisease(BaseModel):
    is_disease: bool


def get_multiple_predictions_no_options(
    fz: FindZebra,
    model: Model,
    n_samples: int,
    query: str,
    n_titles: int = 5,
    temperature: float = 1,
) -> list[str]:
    predictions_raw = []

    for _ in range(n_samples):
        model.complete_add_to_batch_queue(
            f"Which disease are best fitting the following description?\n\nDescription: {query}",
            AnswerFree,
            max_retries=0,  # disable retrying
            model="gpt-4o",
            temperature=temperature,
        )

    predictions_raw = model.complete_execute_queue(AnswerFree, model="gpt-4o")

    # Remove None if any
    predictions_free: list[str] = [
        prediction.disease for prediction in predictions_raw if prediction
    ]

    # Normalize disease names from prediction, using the results of fz
    # get n titles, make model pick the most fitting

    predictions_norm: list[str] = []
    for prediction_free in predictions_free:
        titles: list[str] = [
            title
            for title, _ in fz.search_normalized_batch(
                prediction_free, rows=n_titles, get_raw_title=True
            )
        ]

        class BestTitle(BaseModel):
            title: Literal[tuple(titles)]  # type:ignore

        prompt = (
            f"You are to pick the most general of the following titles that are equal to {prediction_free}.\n"
            f"{titles}"
        )
        prediction_norm = model.complete(prompt, BestTitle).title  # type:ignore

        # print(prediction_free, "closest to", prediction_norm, "of options", titles)

        predictions_norm.append(prediction_norm)

    # make sure that all predictions are represented
    assert len(predictions_norm) == len(
        predictions_free
    ), "lost a prediction in normalization"

    # lower case the predictions
    predictions_norm = [prediction.lower() for prediction in predictions_norm]

    # remove duplicates
    predictions_norm = list(dict.fromkeys(predictions_norm))

    return predictions_norm

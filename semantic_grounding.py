from models import LLM_grounder, FindZebra, ClientOpenAI, Model, ClientLlama
from time import time
from devtools import pprint
from pydantic import BaseModel, Field
from typing import Literal
import torch
from embedding_ada_test import *


class References(BaseModel):
    symptom: str
    references_to_article: list[int]


# fmt: off
class SymptomsList(BaseModel):
    class SymptomSingle(BaseModel):
        symptom_name: str
        is_mentioned_in_article: bool
    symptoms: list[SymptomSingle]
# fmt: on


class YesNo(BaseModel):
    answer: bool


def grounder_retrieve_passages(
    model: Model,
    prediction: str,
    symptoms: str,
) -> tuple[dict[str, list[str]], int]:
    """
    Semantic search retrieval of nearest passages from articles for each symptom
    Returns tuple: {symptom : [list-of-passages]} , int (total length of passages)
    """

    n_passages_threshold = 2  # must have this many or more passages for it to count

    # Get material and embed
    material_object = MaterialEmbedded(
        material=get_multiple_articles_concat(prediction, top_k=5)
    )

    # Split symptoms and embed
    symptoms_sections = symptoms.split(", ")
    symptoms_embeddings = embedder.get_embeddings(symptoms_sections)

    N_RETRIEVED_PASSAGES = 20

    nearest_passages_flat: list[str] = []
    for symptom, embedding in zip(symptoms_sections, symptoms_embeddings):
        # Get k passages from article which are closest to symptom
        retrieved_passages: list[str] = material_object.get_knn(
            embedding, N_RETRIEVED_PASSAGES
        )
        for passage in retrieved_passages:
            # Validate each retrieved passage
            prompt_reference_validation = (
                f"'{passage}'\nAre the above directly related to {symptom}?"
            )
            model.complete_add_to_batch_queue(prompt_reference_validation)
            nearest_passages_flat.append(passage)

    # Execute batched prompts
    filter_valid: list[YesNo | None] = model.complete_execute_queue(YesNo)

    # Restructure flat list of passages to dict [symptom:[validated_passages]]
    symptom_to_validated_passages: dict[str, list[str]] = {
        symptom: [] for symptom in symptoms_sections
    }
    for n, symptom in enumerate(symptoms_sections):
        for e in range(N_RETRIEVED_PASSAGES):
            index = n * N_RETRIEVED_PASSAGES + e
            passage_validation: YesNo | None = filter_valid[index]
            if passage_validation:
                if passage_validation.answer:
                    # Save validated passage to dict under symptom
                    symptom_to_validated_passages[symptom].append(
                        nearest_passages_flat[index]
                    )

    # Remove all symptoms without any passages
    symptom_to_validated_passages = {
        symptom: passages
        for symptom, passages in symptom_to_validated_passages.items()
        if len(passages) >= n_passages_threshold
    }

    # Get total length of validated passages
    support_length = sum(
        [
            len(passage)
            for passages in symptom_to_validated_passages.values()
            for passage in passages
        ]
    )

    return symptom_to_validated_passages, support_length


# %%

if __name__ == "__main__":

    client = ClientOpenAI()
    model = Model(client)

    symptoms: str = (
        "proteinuria (intensity: trace to mild), striae, erythrocyte sedimentation rate (result: elevated) glomerular filtration rate, reduced, alpha 2 globulin level (result: elevated), triglyceride level (result: elevated), serum protein level (result: reduced), increased urinary frequency, erythema of the skin, localized (scaly surface: yes) serum albumin level (result: reduced), sodium level (result: reduced), blood urea nitrogen level (result: elevated) transaminase levels, elevated, focal segmental glomerulosclerosis nephrosclerosis, uric acid level (result: elevated)"
    )

    predictions = [
        "Systemic Lupus Erythematosus (SLE)",
        "Focal segmental glomerulosclerosis (FSGS)",  # <- ground truth
        # "Systemic sclerosis (renal crisis)",
        # "Sjogren's syndrome",
    ]

    for prediction in predictions:
        _, n_passages = grounder_retrieve_passages(model, prediction, symptoms)
        print(prediction, "n passages", n_passages)

    # for prediction in predictions

    # prompt3 = (
    #     f"You are given multiple predictions from a medical disease classifcation task.\n"
    #     f"You are to evaluate which of the diseases has the most support from the attached material.\n"
    #     f"Weight highly specific and uncommon symptoms more. Focus on the support of the references, not the predicted disease.\n\n"
    #     f"Patients symptoms:\n"
    #     f"{symptoms}\n\n"
    # )

    # for prediction, sup in zip(predictions, support):
    #     sup: dict
    #     prompt3 += f"Literature supporting prediction '{prediction}':\n"
    #     # prompt3 += f"Literature supporting prediction A:\n"
    #     prompt3 += ", ".join([val for values in sup.values() for val in values])
    #     prompt3 += f"\n\n\n"

    # print("Final prompt")
    # print(prompt3)

    # # %%

    # class FinalAnswer(BaseModel):
    #     reasoning: str
    #     disease: Literal[tuple(predictions)]  # type: ignore

    # t0 = time()
    # predictions_results: list[str] = []
    # for _ in range(5):
    #     # verdict_object: FinalAnswer = model.complete(prompt3, FinalAnswer, max_retries=1)  # type: ignore
    #     model.complete_add_to_batch_queue(prompt3, FinalAnswer, max_retries=1)  # type: ignore
    # final_predictions = model.complete_execute_queue(FinalAnswer)
    # for pred in final_predictions:
    #     print("-")
    #     print(pred)
    #     if pred:
    #         predictions_results.append(pred.disease)
    # print(f"Time taken for final evaluation: {time() - t0:.2f} seconds")

    # # Find the most common prediction
    # verdict: str = max(set(predictions_results), key=predictions_results.count)

    # print("-")

    # print("Ground truth:", answer)
    # if verdict == answer:
    #     print("Ensemble correct", verdict)
    #     n_success += 1
    # else:
    #     print("Ensemble incorrect", verdict)
    #     n_failures += 1

    # longest_support_prediction: str = ""
    # max_support_length: int = 0
    # for prediction, sup in zip(predictions, support):
    #     support_length = sum(
    #         [
    #             len(passage)
    #             for symptoms_passages in sup.values()
    #             for passage in symptoms_passages
    #         ]
    #     )

    #     if support_length > max_support_length:
    #         max_support_length = support_length
    #         longest_support_prediction = prediction

    # if longest_support_prediction == answer:
    #     print("Simple correct", longest_support_prediction)
    #     n_success_simple += 1
    # else:
    #     print("Simple incorrect", longest_support_prediction)
    #     n_failures_simple += 1

    # total_times.append(int(time() - total_time_start))
    # guesses.append(verdict)
    # guesses_simple.append(longest_support_prediction)

    # print("Embedder token usage:", embedder.tokens_used)
    # client.print_usage()

    # # end of for loop
    # print("successes", n_success)
    # print("failures", n_failures)
    # print("Guesses:", guesses)
    # print("-")
    # print("successes simple", n_success_simple)
    # print("failures simple", n_failures_simple)
    # print("Guesses simple:", guesses_simple)
    # print("-")
    # print("total times", total_times)
    # # Save values to a file
    # with open("grounder_benchmark_semanctic_search_1.txt", "w") as file:
    #     file.write(f"Successes: {n_success}\n")
    #     file.write(f"Failures: {n_failures}\n")
    #     file.write(f"Guesses: {guesses}\n")
    #     file.write(f"Total times: {total_times}\n")
    #     file.write(f"Simple successes: {n_success_simple}\n")
    #     file.write(f"Simple failures: {n_failures_simple}\n")
    #     file.write(f"Simple guesses: {guesses_simple}\n")

    # except Exception as e:
    # import traceback

    # print("An exception occurred:", e)
    # traceback.print_exc()
    # print("Continuing benchmark...")

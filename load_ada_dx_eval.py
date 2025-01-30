# %%
import pandas as pd
from typing import List, Dict, Tuple

# %%


def load_jsonl(file_path):
    """
    Loads a JSONL (JSON Lines) file into a pandas DataFrame.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the JSONL file.
    """
    df = pd.read_json(file_path, lines=True)
    return df


def load_dataset() -> Tuple[List[str], List[str]]:
    df = load_jsonl("ada_dx_eval/ada_dx_eval.jsonl")
    X = df.pathological_symptoms.to_list()
    y = df.confirmed_diagnoses.to_list()
    return X, y


def load_dataset_extended() -> Tuple[List[Dict[str, str | int]], List[str]]:
    """
    Output: list of dicts containing the following columns:
    "sex", "age", "pathological_symptoms", "non-pathological_symptoms"
    """
    df = load_jsonl("ada_dx_eval/ada_dx_eval.jsonl")
    y = df.confirmed_diagnoses.to_list()
    df_X = df[["sex", "age", "pathological_symptoms", "non-pathological_symptoms"]]
    return df_X.to_dict(orient="records"), y  # type:ignore


X, y = load_dataset_extended()
X[1]

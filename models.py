import os
import sys

if (
    os.name == "posix" and not sys.platform == "darwin"
):  # posix indicates a Unix-like OS, excluding macOS
    os.environ["HF_HOME"] = "/work3/s174032/speciale/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import re
from pydantic import BaseModel, Field, model_validator, ValidationInfo
from typing import Callable, List, Set, Tuple, Dict, Optional, Any
from typing_extensions import Self
from openai import OpenAI
from pydantic.functional_validators import field_validator
import requests
import difflib
from enum import Enum
from typing import Annotated
from pydantic import AfterValidator
from abc import ABC, abstractmethod
import textwrap
import json
from collections import Counter
import code
from typing import Type

# from datasets import load_dataset
from openai import OpenAI, AsyncOpenAI
import asyncio
import random
import sys
import time
import threading
import outlines
from pydantic import ValidationError
from time import time

# import llama_cpp
# from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import signal
import re

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    LogitsProcessorList,
)
import outlines
from pydantic import BaseModel, model_validator
from transformers import BitsAndBytesConfig
import torch
import pandas as pd

OPENAI_MODEL = "gpt-4o-mini"


def get_openai_token():
    """
    Reads OpenAI API from local file
    """
    with open("openai_api.txt", "r") as file:
        api_key = file.read().strip()
    return api_key


def get_huggingface_token():
    """
    Reads huggingface token from local file
    """
    with open("hf_api.txt", "r") as file:
        api_key = file.read().strip()
    return api_key


def get_findzebra_token():
    """
    Reads huggingface token from local file
    """
    with open("fz_api.txt", "r") as file:
        api_key = file.read().strip()
    return api_key


def get_fz_corpus() -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        "findzebra/corpus", cache_dir=None
    )  # specify the appropriate dataset name
    corpus = dataset["train"][  # type: ignore
        "title"
    ]  # specify the correct split and column name # type: ignore
    return corpus


def load_fz_dataset() -> Tuple[List[str], List[str]]:
    df = pd.read_csv("fz_dataset_w_titles.csv")
    X = df.question.to_list()
    y = df.title.to_list()
    return X, y


# class Spinner:
#     busy = False
#     delay = 0.1

#     @staticmethod
#     def spinning_cursor():
#         while 1:
#             for cursor in "|/-\\":
#                 yield cursor

#     def __init__(self, message="", delay=None):
#         self.spinner_generator = self.spinning_cursor()
#         self.message = message
#         if delay and float(delay):
#             self.delay = delay

#     def spinner_task(self):
#         while self.busy:
#             print("", self.message, next(self.spinner_generator), end="\r", flush=True)
#             time.sleep(self.delay)

#     def __enter__(self):
#         self.busy = True
#         threading.Thread(target=self.spinner_task).start()

#     def __exit__(self, exception, value, tb):
#         print(f"\033[92m{self.message} - DONE\033[0m")  # final flush
#         self.busy = False
#         time.sleep(self.delay)
#         if exception is not None:
#             return False


class Client:

    @abstractmethod
    def complete(
        self,
        prompt: str,
        return_model: Any,
        *,
        verbose: bool = False,
        temperature: float = -1.0,
        max_new_tokens: int = 500,
    ) -> Any: ...

    @abstractmethod
    def complete_batch(
        self,
        prompts: list[str],
        return_model: type[BaseModel],
        *,
        verbose: bool = False,
        temperature: float = -1.0,
        max_new_tokens: int = 500,
        model="",
    ) -> list[Any]: ...


class ClientOpenAI(Client):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_simul_calls: int = 1000,
        api_key: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = OpenAI(api_key=api_key)
        # self._async_client = AsyncOpenAI()
        self.usage: float = 0  # usage in cash money
        self.max_simul_calls = max_simul_calls

    def _track_usage(self, usage):
        usage_in = usage.prompt_tokens
        usage_out = usage.completion_tokens
        self.usage += self._calculate_price(usage_in, usage_out, self._model)

    def _reset_usage(self):
        self.usage_in = 0
        self.usage_out = 0

    def print_usage(self):
        print(f"Usage {self.usage:.2f}$")

    def _calculate_price(self, usage_in, usage_out, model):
        # prices in $/1M tokens
        models_and_prices = {
            "gpt-4o-mini": [
                0.150,
                0.6,
            ],
            "gpt-4o": [
                2.50,
                10.0,
            ],
        }

        price = (usage_in / 1_000_000) * models_and_prices[model][0] + (
            usage_out / 1_000_000
        ) * models_and_prices[model][1]

        return price

    async def _complete_async(
        self, prompt: str, temperature: float, return_model: Any, model: str
    ) -> str | None:

        async with self.completion_semaphore:
            async with AsyncOpenAI(api_key=self._api_key) as client:
                # client = AsyncOpenAI()
                completion = await client.beta.chat.completions.parse(
                    model=model,
                    response_format=return_model,
                    messages=[
                        {"role": "system", "content": "You are a medical expert."},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=30,
                    temperature=temperature,
                )

        async with self.usage_tracker_semaphore:
            self._track_usage(completion.usage)

        return completion.choices[0].message.content

    async def _gather_complete_async(
        self, prompts, temperature, return_model, model
    ) -> list[str | None]:

        self.usage_tracker_semaphore = asyncio.Semaphore(1)
        self.completion_semaphore = asyncio.Semaphore(self.max_simul_calls)

        tasks = [
            self._complete_async(prompt, temperature, return_model, model)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def complete_batch(
        self,
        prompts: list[str],
        return_model: type[BaseModel],
        *,
        verbose: bool = False,
        temperature: float = -1.0,
        max_new_tokens: int = 500,  # TODO
        model: str = "gpt-4o-mini",
    ) -> list[str | None]:
        # concurrent batch completion

        if not model:
            model = "gpt-4o-mini"

        if temperature == -1.0:
            temperature = 1.0  # default

        completions = asyncio.run(
            self._gather_complete_async(prompts, temperature, return_model, model)
        )

        return completions

        # # Sequential batch
        # return [
        #     self.complete(
        #         prompt, return_model, verbose=verbose, temperature=temperature
        #     )
        #     for prompt in prompts
        # ]

    def complete(
        self,
        prompt: str,
        return_model: Any,
        *,
        verbose: bool = False,
        temperature: float = 1,
        max_new_tokens: int = 500,
        model="gpt-4o-mini",
    ) -> Any:

        if not model:
            model = "gpt-4o-mini"

        if temperature == -1.0:
            temperature = 1.0  # default

        # OpenAI api call
        completion = self._client.beta.chat.completions.parse(
            model=model,
            response_format=return_model,
            messages=[
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": prompt},
            ],
            timeout=30,
            temperature=temperature,
        )

        result = completion.choices[0].message.content

        self._track_usage(completion.usage)

        # save token usage
        # self.track_usage(completion.usage)
        return result


class Quantization(Enum):
    FP = "fp"
    FOUR_BIT = "4b"
    EIGHT_BIT = "8b"


class ClientLlama(Client):
    def __init__(
        self,
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        is_instruct=True,
        download_model=True,
        quantization: None | Quantization = Quantization.EIGHT_BIT,
    ) -> None:
        """
        Initializes local Llama model w. structured output
        Requires ~50GB of VRAM
        """
        self._model_id = model_id
        self._is_instruct = is_instruct
        print("Using model", self._model_id)
        os.environ["HF_HOME"] = "/work3/s174032/speciale/"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        os.system("nvidia-smi")

        # assert torch.cuda.device_count() > 1, "Less than two GPUs available"

        # model_id = "meta-llama/Llama-3.1-8B-Instruct"
        # model_id = "meta-llama/Llama-3.3-70B-Instruct"
        model_id = self._model_id
        token = get_huggingface_token()

        # Load tokenizer and model
        if quantization == Quantization.FOUR_BIT:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == Quantization.EIGHT_BIT:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            # model_id, {"padding_side": "left"}, skip_special_tokens=True, token=token
            model_id,
            padding_side="left",
            skip_special_tokens=True,
            token=token,
        )

        # Make pipeline for inference
        self.generator = pipeline(
            "text-generation",
            model_kwargs={"torch_dtype": "auto"},
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def print_usage(self):
        print("Local model - no token tracking")

    def make_prompt_template(self, prompt: str):
        """
        Creates a prompt template in the openai chat-gpt style for use with Llama
        """
        return [
            {
                "role": "system",
                "content": "You are a medical expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def complete(
        self,
        prompt: str,
        return_model: Any,
        *,
        verbose: bool = False,
        temperature: float = -1.0,
        max_new_tokens: int = 500,
        model="",
    ) -> str:
        """
        Constrained completion using Llama
        """

        if temperature == -1.0:
            temperature = 0.8  # default

        # Create contrained tokenizer and logits_processor
        outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)  # type: ignore
        logits_processor = outlines.processors.JSONLogitsProcessor(  # type: ignore
            return_model,
            outlines_tokenizer,
        )

        # Make prompt template in openai style
        if self._is_instruct:
            prompt_template = self.make_prompt_template(prompt)
        else:
            prompt_template = prompt + "\nAnswer as json: "  # skip chat template

        # Create structured output
        generation = self.generator(
            prompt_template,
            logits_processor=LogitsProcessorList([logits_processor]),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if not self._is_instruct:
            return generation[0]["generated_text"][len(prompt) :]  # type: ignore
        return generation[0]["generated_text"][-1]["content"]  # type: ignore

    def complete_free(
        self,
        prompt: str,
        *,
        verbose: bool = False,
        temperature: float = -1.0,
        max_new_tokens: int = 500,
        model="",
    ) -> str:
        """
        Constrained completion using Llama
        """

        if temperature == -1.0:
            temperature = 0.8  # default

        # Make prompt template in openai style
        if self._is_instruct:
            prompt_template = self.make_prompt_template(prompt)
        else:
            prompt_template = prompt + "\nAnswer as json: "  # skip chat template

        # Create structured output
        generation = self.generator(
            prompt_template,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if not self._is_instruct:
            return generation[0]["generated_text"][len(prompt) :]  # type: ignore
        return generation[0]["generated_text"][-1]["content"]  # type: ignore

    def complete_batch(
        self,
        prompts: list[str],
        return_model: type[BaseModel],
        *,
        verbose: bool = False,
        temperature: float = -1.0,
        max_new_tokens: int = 500,
        model="",
        batch_size=10,
    ) -> list[Any]:
        """
        Structured completion with huggingface pipeline mananging optimal batch completion
        """

        if temperature == -1.0:
            temperature = 0.8  # default

        # ignoring the following

        outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)  # type: ignore
        logits_processor = outlines.processors.JSONLogitsProcessor(  # type: ignore
            return_model, outlines_tokenizer
        )

        # Convert all prompts to chat template
        prompt_templates = [self.make_prompt_template(prompt) for prompt in prompts]

        completions = self.generator(
            prompt_templates,
            logits_processor=LogitsProcessorList([logits_processor]),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            batch_size=batch_size,
        )

        if not completions:
            return []

        return [
            completion[0]["generated_text"][-1]["content"]  # type:ignore
            for completion in completions
        ]

        # texts = self.tokenizer.apply_chat_template(
        #     prompt_templates, add_generation_prompt=True, tokenize=False
        # )

        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # Set a padding token
        # inputs = self.tokenizer(texts, padding="longest", return_tensors="pt")
        # inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        # # Generate responses in one forward pass
        # generation_outputs = self.model.generate(
        #     **inputs,
        #     logits_processor=logits_processor_list,
        #     do_sample=True,
        #     temperature=temperature,
        #     max_new_tokens=max_new_tokens,
        # )

        # prependix = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a medical expert.user\n\nassistant\n\n"

        # # Decode each generation
        # return_values = []
        # for n, (prompt, output) in enumerate(zip(prompts, generation_outputs)):
        #     output_str = self.tokenizer.decode(output, skip_special_tokens=True)
        #     output_str = output_str[len(prependix) + len(prompt) :]
        #     return_values.append(output_str)
        # return return_values


class Model:
    _client: Client

    def __init__(self, client: Client) -> None:
        self._client = client
        self._queue: list[str] = []

    def complete(
        self,
        prompt: str,
        return_model: Type[BaseModel],
        *,
        verbose: bool = False,
        max_retries: int = 1,
        temperature: float = -1.0,
        validation_context: dict[str, Any] = {},
        soft_retry: bool = False,
        max_new_tokens=500,
        model="",
    ):
        # Process the prompt structure
        prompt_decorated = f"{prompt}"  # i.e. no decorations

        # Get response from llm client
        response_raw: str = self._client.complete(
            prompt_decorated,
            return_model,
            verbose=verbose,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # # Test the retry logic
        # if max_retries == 3:
        #     response_raw += "{' }}}"

        try:
            # Test response is valid json
            _ = json.loads(response_raw)
            # Create pydantic model
            response = return_model.model_validate_json(
                response_raw,
                context=validation_context,
            )
        except Exception as e:
            # Ask model to fix the invalid answer
            if verbose:
                print(e, "max retries:", max_retries)
                print("")
            if max_retries > 0:
                prompt_retry = (
                    f"Correct the following JSON response, based on the errors given below:\n\n"
                    f"JSON:\n{response_raw}\n\nExceptions:\n{e}"
                )
                response = self.complete(
                    prompt_retry, return_model, max_retries=max_retries - 1
                )
            else:
                print("All retries failed.")
                response = None  # TODO maybe fill with neutral values

        return response

    def complete_add_to_batch_queue(
        self,
        prompt: str,
        return_model=None,  # well, does not matter since it is inputted in the execute function.
        *,
        verbose: bool = False,
        max_retries: int = 1,
        temperature: float = -1.0,
        validation_context: dict[str, Any] = {},
        soft_retry: bool = False,
        max_new_tokens=500,
        model="",
    ):
        self._queue.append(prompt)

    def complete_execute_queue(
        self, return_model: Type[BaseModel], model="gpt-4o-mini", temperature=-1
    ):
        print("Executing queue of length", len(self._queue))
        try:
            to_return = self.complete_batch(
                self._queue, return_model, model=model, temperature=temperature
            )
        except Exception as e:
            self._queue = []
            raise e

        self._queue = []  # empty queue
        return to_return

    def complete_batch(
        self,
        prompts: list[str],
        return_model: Type[BaseModel],
        *,
        verbose: bool = False,
        max_retries: int = 1,
        temperature: float = -1.0,
        validation_context: dict[str, Any] = {},
        soft_retry: bool = False,
        max_new_tokens=500,
        model="",
    ):
        responses_raw = []
        for retry_counter in range(max_retries + 1):
            try:
                responses_raw = self._client.complete_batch(
                    prompts,
                    return_model,
                    verbose=verbose,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    model=model,
                )
            except Exception as e:
                print(repr(e))
                if retry_counter == max_retries:
                    raise e
                print("Retrying...")

        responses: list[Any] = []
        prompts_retry_batch: list[str] = []
        for response_raw in responses_raw:
            try:
                # Test response is valid json
                _ = json.loads(response_raw)
                # Create pydantic model
                response = return_model.model_validate_json(
                    response_raw,
                    context=validation_context,
                )
                responses.append(response)
            except Exception as e:
                # Ask model to fix the invalid answer
                if verbose:
                    print(e, "max retries:", max_retries)
                    print("")
                if max_retries > 0:
                    prompt_retry = (
                        f"Correct the following JSON response, based on the errors given below:\n\n"
                        f"JSON:\n{response_raw}\n\nExceptions:\n{e}"
                    )
                    prompts_retry_batch.append(prompt_retry)

        if prompts_retry_batch:
            if verbose:
                print(
                    "Max retries", max_retries, "n_retrying", len(prompts_retry_batch)
                )
            responses += self.complete_batch(
                prompts_retry_batch, return_model, max_retries=max_retries - 1
            )

        return responses


class LlamaStructured:
    def __init__(self, *, max_retries=3) -> None:
        self.max_retries = max_retries
        os.environ["HF_HOME"] = "/work3/s174032/speciale/"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        os.system("nvidia-smi")

        # assert torch.cuda.device_count() > 1, "Less than two GPUs available"

        # model_id = "meta-llama/Llama-3.1-8B-Instruct"
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        token = ""  # moved to file and this object is depricated

        # Load tokenizer and model
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        # quantization_config = BitsAndBytesConfig(
        #     load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        # )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, {"padding_side": "left"}, skip_special_tokens=True, token=token
        )

        # Make pipeline for inference
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def make_prompt_template(self, prompt: str):
        return [
            {
                "role": "system",
                "content": "You are a medical expert.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def complete_batch_free_form(
        self,
        prompts: list[str],
        *,
        verbose: bool = False,
        max_retries: int = 3,  # TODO figure how to implement retries for this particular model
        temperature: float = 1.0,
        validation_context: dict[str, Any] = {},
        max_new_tokens=1000,
    ) -> list[Any]:
        # add instructions for formatting
        prompts = [
            prompt
            + '\nPlease structure the prediction like the following example: {"disease": "*insert predicted disease*"}'
            for prompt in prompts
        ]

        for prompt in prompts:
            if verbose:
                print(prompt)

        # Tokenize inputs and batch them
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        # Generate responses in one forward pass
        generation_outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=1,
            max_new_tokens=max_new_tokens,
        )

        # Decode and print each generation
        n_errors = 0
        return_values = []
        for n, (prompt, output) in enumerate(zip(prompts, generation_outputs)):
            output_str = self.tokenizer.decode(output, skip_special_tokens=True)[
                len(prompt) :
            ]
            if verbose:
                print("output:", output_str)
            return_values.append(output_str)

        print("Complete:", n_errors, "retried sequentally out of", len(prompts))
        return return_values

    def complete_batch(
        self,
        prompts: list[str],
        return_model: Any,
        *,
        verbose: bool = False,
        max_retries: int = 3,  # TODO figure how to implement retries for this particular model
        temperature: float = 1.0,
        validation_context: dict[str, Any] = {},
    ) -> list[Any]:
        # add instructions for formatting
        prompts = [
            prompt
            + "\nFormat the output as the following object: {return_model.model_json_schema()}{prev_error_prompt}"
            for prompt in prompts
        ]

        for prompt in prompts:
            if verbose:
                print(prompt)

        outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)  # type: ignore
        logits_processor = outlines.processors.JSONLogitsProcessor(  # type: ignore
            return_model,
            outlines_tokenizer,
        )

        logits_processor_list = LogitsProcessorList([logits_processor])

        # Tokenize inputs and batch them
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        # Generate responses in one forward pass
        generation_outputs = self.model.generate(
            **inputs,
            logits_processor=logits_processor_list,
            do_sample=True,
            temperature=temperature,
            top_p=1,
            max_new_tokens=1000,
        )

        # Decode and print each generation
        n_errors = 0
        return_values = []
        for n, (prompt, output) in enumerate(zip(prompts, generation_outputs)):
            try:
                output_str = self.tokenizer.decode(output, skip_special_tokens=False)[
                    len(prompt) :
                ]
                if verbose:
                    print("output:", output_str)
                output_model = return_model.model_validate_json(
                    output_str, context=validation_context
                )
                return_values.append(output_model)
            except Exception as e:
                n_errors += 1
                print(repr(e))
                # # print(f"Error on {n}:", e)
                # # If any prompts fail, redo one at a time TODO batch these as well
                # if max_retries > 0:
                #     del inputs  # type: ignore
                #     torch.cuda.empty_cache()  # Clear CUDA memory cache
                #     prompt += f"\nPlease improve on the previous error: {e}"
                #     output_model = self.complete(
                #         prompt,
                #         return_model,
                #         max_retries=max_retries,
                #         validation_context=validation_context,
                #     )
                #     return_values.append(output_model)
        print("Complete:", n_errors, "retried sequentally out of", len(prompts))
        return return_values

    def _complete_raw(
        self,
        prompt: list[dict[str, str]],
        return_model: Any,
        *,
        verbose: bool = False,
        max_retries: int = 3,
        temperature: float = 1.0,
        validation_context: dict[str, Any] = {},
        soft_retry: bool = False,
        max_new_tokens=500,
    ) -> str:
        # Constrain sampling using outlines logits_processor
        outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)  # type: ignore
        logits_processor = outlines.processors.JSONLogitsProcessor(  # type: ignore
            return_model,
            outlines_tokenizer,
        )

        # structure the output
        generation = self.generator(
            prompt,
            logits_processor=LogitsProcessorList([logits_processor]),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # top_p=0.8,
        )
        if verbose:
            print("generation", generation)
        return generation[0]["generated_text"][-1]["content"]  # type: ignore

    def complete(
        self,
        prompt_txt: str,
        return_model: Any,
        *,
        verbose: bool = False,
        max_retries: int = 3,
        temperature: float = 1.0,
        validation_context: dict[str, Any] = {},
        soft_retry: bool = False,
        max_new_tokens=500,
    ):
        # Generate free answer
        prev_errors: list[str] = []
        for _ in range(max_retries + 1):
            # print("---")

            # Concat previous errors to the prompt
            if prev_errors:
                prev_error_prompt = (
                    f"\nDo not repeat the previous errors: {prev_errors}."
                )
            else:
                prev_error_prompt = ""

            prompt_object = self.make_prompt_template(
                f"{prompt_txt}. Format the output as the following object: {return_model.model_json_schema()}{prev_error_prompt}"
            )

            raw_response = self._complete_raw(
                prompt_object,
                return_model,
                verbose=verbose,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            try:
                response = return_model.model_validate_json(
                    raw_response, context=validation_context
                )
                if verbose:
                    print(response, end=", ")
                return response
            except Exception as e:
                if verbose:
                    print(e)

                # Retry without rerunning the full prompt
                prompt_soft_retry = f"Validation Error found:\n{e}\nStructure the object correctly, fix the errors.\nFormat the output as the following object: {return_model.model_json_schema()}"
                print("calling", prompt_soft_retry)
                raw_response_soft_retry = self._complete_raw(
                    self.make_prompt_template(prompt_soft_retry),
                    return_model,
                )

                try:
                    response = return_model.model_validate_json(
                        raw_response, context=validation_context
                    )
                    return response
                except:
                    print("Soft retry failed")

                prev_errors.append(repr(e))
                continue
            break


class LocalClient:
    class Completion:
        class Usage:
            def __init__(self):
                self.total_tokens = 0
                self.prompt_tokens = 0
                self.completion_tokens = 0

        def __init__(self):
            self.usage = self.Usage()

    def __init__(self, complete_fn) -> None:
        self.complete_fn = complete_fn

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create_with_completion(
                self,
                model: Any,
                response_model: Any,
                messages: List[Dict[str, str]],
                max_retries: int,
                timeout: float,
                temperature: float,
            ) -> Tuple[Any, "LocalClient.Completion"]:
                # Extract the prompt from the provided messages
                prompt = messages[1]["content"]
                object = self.parent.complete_fn(prompt, response_model)

                # Create a Completion object that mimics OpenAI's usage structure
                completion = LocalClient.Completion()

                return object, completion


class LLM:
    """
    Parent object for all LLM models
    - add completion functions for calling openai via instructor - both sequential and concurrent
    - adds token usage tracking capabilities
    """

    model = OPENAI_MODEL
    usage_total = 0
    usage_in = 0
    usage_out = 0
    client = None
    async_client = None
    usage_tracker_semaphore = asyncio.Semaphore(1)

    @classmethod
    def set_client(cls, client):
        cls.client = client

    def track_usage(self, usage):
        self.usage_total += usage.total_tokens
        self.usage_in += usage.prompt_tokens
        self.usage_out += usage.completion_tokens

    def reset_usage(self):
        self.usage_total = 0
        self.usage_in = 0
        self.usage_out = 0

    def print_usage(self):
        models_and_prices = {
            # in $/1M tokens
            "gpt-4o-mini": [
                0.150,
                0.6,
            ],  # $0.150 / 1M input tokens, $0.600 / 1M output tokens
            "gpt-4o": [
                2.50,
                10.0,
            ],  # $5.00 / 1M input tokens, $15.00 / 1M output tokens
            "o1-mini": [
                3.0,
                12.0,
            ],  # $5.00 / 1M input tokens, $15.00 / 1M output tokens
            "o1-preview": [
                15.0,
                60.0,
            ],  # $5.00 / 1M input tokens, $15.00 / 1M output tokens
            # "o1-preview": [ # NOTE requires tier 5 api access p.t.
            #     15.0,
            #     60.0,
            # ],  # $5.00 / 1M input tokens, $15.00 / 1M output tokens
            # "o1-mini": [
            #     3.0,
            #     12.0,
            # ],  # $5.00 / 1M input tokens, $15.00 / 1M output tokens
        }

        prompt_tokens = self.usage_in
        completion_tokens = self.usage_out
        total_tokens = self.usage_total

        price = (prompt_tokens / 1_000_000) * models_and_prices[self.model][0] + (
            completion_tokens / 1_000_000
        ) * models_and_prices[self.model][1]
        print(
            f"{self.__class__.__name__}: {total_tokens} tokens ({prompt_tokens} prompt + {completion_tokens} completion) costing ${price:.3f}"
        )

    @abstractmethod
    def complete(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_answer(self, phenotypes: str, options: Set[str]) -> str:
        pass

    def create_complete(
        self,
        return_object: Any,
        role_description: str = "You are a medical expert.",
        max_retries: int = 5,
        temperature: float = 1,
        timeout: int = 30,
    ) -> Any:
        """
        Function factory for the llm's 'complete' function
        Input: various inputs for the complete function
        Output: function for completion
        """

        class_name: str = self.__class__.__name__

        if self.model != OPENAI_MODEL:
            print(f"{class_name}, using, {self.model}")

        # define function
        def complete(content: str, validation_context=None):

            try:
                # OpenAI api call
                result, completion = (
                    self.client.chat.completions.create_with_completion(  # type:ignore , ignores that client.chat is not recoqnized
                        model=self.model,
                        response_model=return_object,
                        messages=[
                            {"role": "system", "content": role_description},
                            {"role": "user", "content": content},
                        ],
                        max_retries=max_retries,
                        timeout=timeout,
                        temperature=temperature,
                    )
                )
                # save token usage
                self.track_usage(completion.usage)
                return result

            except InstructorRetryException as e:  # type: ignore
                # ask model to return neutral object (this is some convoluted shit)
                try:
                    # OpenAI api call
                    result, completion = (
                        self.client.chat.completions.create_with_completion(  # type:ignore , ignores that client.chat is not recoqnized
                            model=self.model,
                            response_model=return_object,
                            messages=[
                                {"role": "system", "content": role_description},
                                {
                                    "role": "user",
                                    "content": "Fill out the return model with neutral values e.g. No answer, while still following any pydantic validators etc.",
                                },
                            ],
                            max_retries=3,
                            timeout=30,
                        )
                    )
                    # save token usage
                    self.track_usage(completion.usage)
                    return result
                except InstructorRetryException as e:  # type: ignore
                    import traceback

                    traceback.print_exc()
                    return None  # TODO return neutral object

        return complete


class SimpleAnswer(BaseModel):
    """
    Return object for answering agent. Contains only the concise answer.
    """

    answer: str = Field(
        ...,
        description="Concise answer to the question <200 characters. If there is no answer then explain why.",
    )


class LLM_modular(LLM):

    def __init__(
        self, return_object: Any, role_description: str = "You are a helpful expert."
    ) -> None:
        model = "gpt-4o"
        self.complete = self.create_complete(
            role_description=role_description,
            return_object=return_object,
            max_retries=2,
        )

    def complete(self, prompt: str) -> Any:
        return self.complete(prompt)


class LLM_answerer_simple(LLM):
    def __init__(self) -> None:
        self.complete = self.create_complete(
            role_description="You are a QA agent that answers medical questions concisely. Given the following description of a patients phenotypes, pick the disease from the options that is most likely to be the root course.",
            return_object=SimpleAnswer,
            max_retries=2,
        )

    def get_answer(self, phenotypes, options) -> str:
        prompt = f"What disease is the course of the following symptoms?\n{phenotypes}"
        return self.complete(prompt).answer


class Reference(BaseModel):
    _return_suggestions: bool = True
    title: str = Field(..., description="Title of referenced material.")
    reference: str = Field(..., description="Exact citation from the material.")

    # String lookup function
    def _string_lookup(self, reference: str, material: str) -> bool:
        # reference = re.sub(r"[^\w\s]", "", reference) # remove punctuation
        # material = re.sub(r"[^\w\s]", "", material)

        is_matching = bool(re.search(re.escape(reference), material, re.IGNORECASE))
        is_matching_except_final_punctuation = bool(
            re.search(re.escape(reference[:-7]), material, re.IGNORECASE)
        )
        reference_alpha = re.sub(r"\W+", "", reference)
        material_alpha = re.sub(r"\W+", "", material)
        is_matching_alphanumeric = bool(
            re.search(re.escape(reference_alpha), material_alpha, re.IGNORECASE)
        )
        return (
            is_matching_alphanumeric
            or is_matching
            or is_matching_except_final_punctuation
        )

    def _find_close_matches(self, reference: str, material: str) -> List[str]:
        """
        Finds close matches of keyword in a corpus of text. Similar to 'Did you mean:'
        """
        chunks = re.split(
            r"(?<=\.)\s+|\n", material
        )  # Adjust splitting for better chunking while keeping punctuation
        suggestions = difflib.get_close_matches(reference, chunks, cutoff=0.1, n=3)

        # sanity testing alignment of search and suggestings
        for suggestion in suggestions:
            assert self._string_lookup(
                suggestion, material
            ), f"Suggestion not found in material: {suggestion}"
        return suggestions

    @model_validator(mode="after")
    def validate_sources(self, info: ValidationInfo) -> "Reference":
        if not info.context or not self.reference:
            self.reference = "No reference or no validation context"
            return self
        material = info.context.get("article", None)
        if self.reference in {"No support", "no support", "No support.", "no support"}:
            return self
        if not self._string_lookup(self.reference, material):
            if self._return_suggestions:
                suggestions = self._find_close_matches(self.reference, material)
                raise ValueError(
                    f"The citation '{self.reference}' does not match the attached material. Make sure that it is exactly as in the material or that it says 'No support'. Did you mean any of these matching quotes?: {suggestions}"
                )
            else:
                raise ValueError(
                    f"'{self.reference}' not found in the attached material."
                )
        return self


class LLM_synonym_merger(LLM):
    model = "gpt-4o-mini"

    class Answer(BaseModel):
        optimized_diseases: list[str]

    def __init__(self) -> None:
        self.complete = self.create_complete(self.Answer)

    def merge_terms(self, diseases: list[str]) -> list[str]:
        prompt = textwrap.dedent(
            f"""
            Optimize the list of disease names by categorizing them into more general terms.
            Ensure synonymous terms are merged without losing any specific details.
            Order by most frequent to least common.
            List of diseases:
            {diseases}
        """.strip()
        )
        return self.complete(prompt).optimized_diseases


class LLM_synonymous_checker(LLM):
    model = "gpt-4o-mini"

    class Answer(BaseModel):
        reasoning: str
        is_synonymous: bool

    def __init__(self) -> None:
        self.complete = self.create_complete(self.Answer, temperature=0)

    def compare(self, term1: str, term2: str) -> Any:
        prompt = textwrap.dedent(
            f"""
            Evaluate whether the two disease names provided are synonymous.
            Assess if both names refer to the same disease entity or condition.

            Term 1:
            {term1}

            Term 2:
            {term2}
        """.strip()
        )
        return self.complete(prompt)


class LLM_grounder_new:

    def __init__(self, client: LlamaStructured) -> None:
        self.client = client

    def complete(self, prediction: str, symptoms: str):
        fz = FindZebra()

        class References(BaseModel):
            symptom: str
            references_to_article: list[int]

        class SymptomsList(BaseModel):
            class SymptomSingle(BaseModel):
                symptom_name: str
                is_mentioned_in_article: bool

            symptoms: list[SymptomSingle]

        print("Grounding", prediction)
        title, material = fz.search_normalized_batch(prediction)[0]
        print("using material", title)

        # Get mentioned symptoms
        prompt = (
            f"List each of the attached symptoms and evaluate whether it is referred to in the attached article (both explicitly or implicitly).\n"
            f"Article: {material}\n\n"
            f"Symptoms: {symptoms}"
        )

        t0 = time()
        symptomslist_object: SymptomsList = self.client.complete(
            prompt, SymptomsList, temperature=0.8
        )  # type: ignore
        end_time = time()
        print(f"Time taken to complete symptoms list: {end_time - t0:.2f} seconds")

        symptoms_list = [
            symptom.symptom_name
            for symptom in symptomslist_object.symptoms
            if symptom.is_mentioned_in_article
        ]
        print("relevant symptoms", symptoms_list)

        print("Now finishing the grounding")
        # Make indexed material
        material_sections = [
            sentence.strip()
            for sentence in material.replace("\n", ".").split(".")
            if sentence.strip()
        ]
        material_indexed = ". ".join(
            [f"[{n}] {section}" for n, section in enumerate(material_sections)]
        )

        all_references: list[References] = []
        t0 = time()
        for symptom in symptoms_list:
            prompt2 = (
                f"From the attached article, find all implicit and explicit references to the symptom below.\n"
                f"Output the name of the symptom as well as the indexes of relevant passages from the article."
                f"Article: {material_indexed}\n\n"
                f"Symptom: {symptom}"
            )
            all_references.append(
                self.client.complete(prompt2, References, temperature=0.8)  # type: ignore
            )
        print(f"Time taken to complete references list: {time() - t0:.2f} seconds")
        print("references raw", all_references)

        # evidence[prediction] = all_references

        collected_references = set()
        for reference in all_references:
            collected_references.update(
                {idx for idx in reference.references_to_article}
            )

        # Unpack index quotes

        all_references_str = " (...) ".join(
            [material_sections[idx] for idx in collected_references]
        )
        print("all references", all_references_str)
        evidence[prediction] = all_references_str  # type: ignore


class LLM_grounder(LLM):
    class Match(BaseModel):
        class Rating(Enum):
            NO_RATING = -1
            COMPLETELY_IRRELEVANT = 0
            RELEVANT_WITH_MAJOR_ASSUMPTIONS = 1
            RELEVANT_WITH_MINOR_ASSUMPTIONS = 2
            PERFECT_CORRELATION = 3

        symptom: str = Field(
            ..., description="Symptom that relates to the quotes from literature"
        )
        reference_literature: Reference = Field(
            ...,
            description="Quote from literature that relates to symptom.",
        )
        reasoning: str = Field(
            ...,
            description="Reasoning for the match",
        )
        rating: Rating = Field(
            ..., description="Correlation and relevance of the match."
        )

    class Symptoms(BaseModel):
        symptoms: list[str]

    def __init__(self, client: LlamaStructured) -> None:
        self.client = client

    def complete(
        self, symptoms: str, article: str, threshold=0, verbose: bool = False
    ) -> list[Match]:
        """
        Returns references for symptoms in article
        """

        # Extract symptoms with timing
        symptoms_extracted = self.client.complete(
            f"Extract all present symptoms from the following text: {symptoms}",
            self.Symptoms,
        )

        prompts = [
            textwrap.dedent(
                f"""
                Provide support for a predicted disease by referencing a patient's symptoms with quotes from the attached articles.
                Provide concise and strictly exact quotes. If quote are shorter than the article, then finish it with (...)
                Cover all below symptoms even if they are not supported by the article (just fill in 'No support' as reference)

                Material:
                {article}

                Symptom:
                {symptom}
            """.strip()
            )
            for symptom in symptoms_extracted.symptoms  # type: ignore
        ]

        # Concurrent grounding
        matches = self.client.complete_batch(
            prompts, self.Match, validation_context={"article": article}, verbose=False
        )

        # # Sequential grounding
        # matches = []
        # for prompt in prompts:
        #     match = self.client.complete(
        #         prompt,
        #         self.Match,
        #         validation_context={"article": article},
        #         verbose=verbose,
        #         temperature=0.8,
        #     )
        #     matches.append(match)

        # remove None #TODO rewrite complete to ensure neutral objects worst case
        matches = [match for match in matches if match]

        # remove none ratings
        matches_not_none = [match for match in matches if match.rating]
        # remove bad ratings
        matches_good = [
            match for match in matches_not_none if match.rating.value > threshold
        ]
        # for match in matches_good:
        #     del match.rating

        return matches_good


class LLM_answerer_reference_evaluator(LLM):
    class Prediction(BaseModel):
        reasoning: str = Field(
            ..., description="Evaluation of each disease and their matching symptoms"
        )
        diseases: list[str] = Field(
            ..., description="Reordered predictions from best fitting to worst"
        )

    def __init__(self, n_models=10) -> None:
        self.n_models = n_models
        self.complete = self.create_complete(
            role_description="You are a concise medical expert.",
            return_object=self.Prediction,
            max_retries=2,
        )

    def evaluate(
        self, symptoms: str, all_references: dict[str, Any]
    ) -> "LLM_answerer_reference_evaluator.Prediction":
        prompt = f"""
        You are given multiple predictions from a medical disease classifcation task.
        Each prediction has a list of references between the observed symptoms from the patients journal and typical symptoms for that particular disease.
        You are to evaluate which of the diseases has the most support from the material.
        Weight highly specific and uncommon symptoms more. Do not put too much weight on the ensemble count of each prediction.

        Patients symptoms:
        {symptoms}
        """.strip()

        for n, (prediction, references) in enumerate(all_references.items()):
            prompt += f"""
            Predicted disease {n+1}: >{prediction}<
            Support:
            {references}
            """

        return self.complete(prompt)


class LLM_answerer_multiple_samples(LLM):
    model = "gpt-4o-mini"

    class Answer(BaseModel):
        disease: str = Field(..., description="Predicted disease")

    def __init__(
        self,
        top_k: int = 5,
        n_samples: int = 20,
        temperature: float = 1,
        threshold: float = 0.1,
        with_extra: bool = False,
    ) -> None:
        self.top_k = top_k
        self.n_samples = n_samples
        self.model = "gpt-4o-mini"
        self.threshold = threshold
        self.with_extra = with_extra
        self.complete = self.create_complete(self.Answer)

    def get_answers(self, phenotypes: str, get_counts=False) -> List[str]:
        # TODO: remove get_counts
        # concurrently get completions from answererer model
        predictions = [self.complete(phenotypes) for _ in range(self.n_samples)]

        # extract str
        predictions = [elem.disease for elem in predictions]

        # remove None-type
        predictions = [elem for elem in predictions if elem]

        # remove all 'No answer' entries
        predictions = [pred for pred in predictions if pred != "No answer"]

        return predictions


# Normalize both sentences
def normalize_text(text):
    # Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")

    # Remove HTML tags using regex
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace newlines and multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text


class FindZebra:
    """
    Object for retrieving search results from findzebra.com
    """

    def __init__(self, api_key=""):
        self.fields = "title,display_content,source"
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = get_findzebra_token()

    def search_w_card_stacking(
        self, query: str, answer: str, rows=10
    ) -> List[Tuple[str, str]]:
        search_result: List[Tuple[str, str]] = self.search_normalized_batch(query, rows)
        answer_result: Tuple[str, str] = self.search_normalized_batch(answer, 1)[0]
        index = len(search_result) // 2
        search_result.insert(index, answer_result)
        return search_result

    def search_cui(self, cui: str) -> List[Tuple[str, str]]:
        r = requests.get(
            "https://www.findzebra.com/api/v1/query_cuis",
            params={
                "api_key": self.api_key,
                "cuis": cui,
                "fl": self.fields,
            },
        )
        assert r.status_code == 200
        results = []
        for result in r.json()["response"]["docs"]:
            results.append((result["title"], result["display_content"]))
        return results

    def search_normalized_batch(
        self, query: str, rows: int = 5, get_raw_title=False
    ) -> List[Tuple[str, str]]:
        """
        Input: query (str): search query
        Optional: rows (int): number of results returned
        Returns: list of (title,content) with length @rows
        """
        articles: List[Tuple[str, str]] = []
        results = self.search(query, rows)
        try:
            for result in results:
                if not get_raw_title:
                    title = f"{result['title']} ({result['source']})"
                else:
                    title = result["title"]
                content = normalize_text(result["display_content"])
                articles.append((title, content))
        except Exception as e:
            print("FZ ERROR", e)
            print("^ query was", query)

        while len(articles) < rows:
            articles.append(("No title", "No content"))
        return articles

    def search_normalized(self, query: str, rows: int = 3) -> Tuple[str, List[str]]:
        all_text = ""
        titles = []
        results = self.search(query, rows)
        for result in results:
            title = result["title"]
            titles.append(title)
            content = normalize_text(result["display_content"])
            all_text += f"Title: {title}\n{content}\n\n"  # accumulate all material for final answer
        return all_text, titles

    def search(self, query: str, rows: int = 3) -> List[Dict]:
        """
        Input: search query
        Output: list of dicts (title, content), each being a result
        """
        try:
            r = requests.get(
                "https://www.findzebra.com/api/v1/query",
                params={
                    "api_key": self.api_key,
                    "q": query,
                    "rows": rows,
                    "fl": self.fields,
                },
            )
            assert r.status_code == 200
            return r.json()["response"]["docs"]
        except Exception as e:
            print(f"FZ: retrieval failed for query: {query}.")
            return [
                {"title": "No title", "display_content": "No content"}
            ] * rows  # return neutral object

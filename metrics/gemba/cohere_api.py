import os
import sys
import time
import logging
from termcolor import colored

import cohere
from cohere import ClientV2
import tqdm

from metrics.gemba.errors import (
    ERROR_MAX_TOKENS,
    ERROR_MAX_RETRY,
    ERROR_MAX_TEMPERATURE,
)


class CohereApi:
    def __init__(self, verbose=False):
        self.verbose = verbose

        if "COHERE_API_KEY" in os.environ:
            self.client = ClientV2(api_key=os.environ.get("COHERE_API_KEY"))
        else:
            raise Exception("COHERE_API_KEY not found in environment!")

        logging.getLogger().setLevel(
            logging.CRITICAL
        )  # to suppress all these HTTP INFO log messages

    def request(
        self,
        prompt,
        model,
        parse_response,
        temperature=0,
        answer_id=-1,
        cache=None,
        max_tokens=8192,
    ):
        request = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

        if request in cache and cache[request] is not None and len(cache[request]) > 0:
            answers = cache[request]
        else:
            answers = self.request_api(prompt, model, temperature, max_tokens)
            if isinstance(answers, str):
                cache[request] = []
            else:
                cache[request] = answers

        if isinstance(answers, str):
            error = answers
            return [
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": None,
                    "prompt": prompt,
                    "finish_reason": error,
                    "model": model,
                }
            ]

        parsed_answers = []
        for full_answer in answers:
            finish_reason = full_answer["finish_reason"]
            full_answer = full_answer["answer"]
            answer_id += 1
            answer = parse_response(full_answer)
            if self.verbose or temperature > 0:
                print(
                    f"Answer (t={temperature}): "
                    + colored(answer, "yellow")
                    + " ("
                    + colored(full_answer, "blue")
                    + ")",
                    file=sys.stderr,
                )
            if answer is None:
                continue
            parsed_answers.append(
                {
                    "temperature": temperature,
                    "answer_id": answer_id,
                    "answer": answer,
                    "prompt": prompt,
                    "finish_reason": finish_reason,
                    "model": model,
                }
            )

        # there was no valid answer, increase the temperature parameter and try again
        if len(parsed_answers) == 0:
            return self.request(
                prompt,
                model,
                parse_response,
                temperature=temperature + 1,
                answer_id=answer_id,
                cache=cache,
                max_tokens=max_tokens,
            )

        return parsed_answers

    def request_api(self, prompt, model, temperature=0, max_tokens=8192, max_retry=10):
        if temperature > 10:
            return ERROR_MAX_TEMPERATURE

        curr_retry, response = 0, None
        while curr_retry < max_retry:
            try:
                response = self.call_api(prompt, model, temperature, max_tokens)
                break
            except Exception as e:
                # A frequent error is reaching the API limit for ex.
                curr_retry += 1
                print(
                    colored(
                        f"Error in attempt number {curr_retry}, retrying...", "red"
                    ),
                    file=sys.stderr,
                )
                print(e, file=sys.stderr)
                time.sleep(1)

        if response is None:
            print(
                colored(f"Failed to get a response after {max_retry} retries.", "red"),
                file=sys.stderr,
            )
            return ERROR_MAX_RETRY
        if response == ERROR_MAX_TOKENS:
            return ERROR_MAX_TOKENS

        answer, finish_reason = response
        answers = [{"answer": answer, "finish_reason": finish_reason}]
        return answers

    def call_api(self, prompt, model, temperature, max_tokens):
        if isinstance(prompt, list):
            # check that prompt is a list of dictionaries with role and content
            assert all(
                isinstance(p, dict) for p in prompt
            ), "Prompts must be a list of dictionaries."
            assert all(
                "role" in p and "content" in p for p in prompt
            ), "Prompts must be a list of dictionaries with role and content."

        else:
            prompt = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]

        try:
            response = self.client.chat(
                model=model,
                temperature=temperature,
                messages=prompt,
                max_tokens=max_tokens,
            )
        except (
            cohere.errors.bad_request_error.BadRequestError,
            cohere.errors.unprocessable_entity_error.UnprocessableEntityError,
        ) as err:
            if "too many tokens" in err.body["message"]:
                return ERROR_MAX_TOKENS
            raise err

        if response.finish_reason == "MAX_TOKENS":
            return ERROR_MAX_TOKENS

        if response.finish_reason != "COMPLETE":
            logging.warning(f"Finish reason: {response.finish_reason}")
            raise Exception(f"Unexpected finish reason: {response.finish_reason}")

        return response.message.content[0].text, response.finish_reason

    def bulk_request(self, df, model, parse_response, cache, max_tokens=8192):
        answers = []
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stderr):
            prompt = row["prompt"]
            parsed_answers = self.request(
                prompt, model, parse_response, cache=cache, max_tokens=max_tokens
            )
            answers += parsed_answers
        return answers

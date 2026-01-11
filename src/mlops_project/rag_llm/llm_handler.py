import os
import sys
from contextlib import contextmanager

from llama_cpp import Llama


@contextmanager
def suppress_stderr():
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


class LLMHandler:
    def __init__(self, model_path: str, n_gpu_layers: int, context_length: int):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=context_length,
            logits_all=False,
            verbose=False,
        )

    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a wildfire emergency response planner."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return output["choices"][0]["message"]["content"].strip()

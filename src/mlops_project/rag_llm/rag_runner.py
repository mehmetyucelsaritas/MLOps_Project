from mlops_project.rag_llm.llm_handler import LLMHandler
from mlops_project.rag_llm.prompt_builder import PromptBuilder


class RAGPipeline:
    def __init__(self, model_path: str, n_gpu_layers: int, context_length: int):
        self.llm_handler = LLMHandler(model_path, n_gpu_layers, context_length)

    def run(self, user_input: str, retrieved_chunks: list[dict], max_tokens: int = 2048) -> str:
        prompt = PromptBuilder.build_wildfire_action_plan_prompt(user_input, retrieved_chunks)
        return self.llm_handler.generate_response(prompt, max_tokens=max_tokens)

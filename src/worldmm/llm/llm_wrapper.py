"""Thin wrapper around the OpenRouter LLM client.

Historical note: this module used to dispatch between OpenAI / Qwen3VL / OpenRouter
backends. All call sites now go through OpenRouter, so the wrapper just instantiates
``OpenRouterModel`` directly. The ``provider`` keyword is preserved for backward
compatibility with existing call sites but is ignored.
"""
from typing import Any, Dict, List, Optional, Union


class LLMModel:
    def __init__(self, model_name: str, *, provider: Optional[str] = None, **kwargs):
        from .openrouter import OpenRouterModel

        self.model_name = model_name
        self.provider = "openrouter"
        self.model = OpenRouterModel(model_name=model_name, **kwargs)

    def generate(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs) -> Any:
        return self.model.generate(prompt, **kwargs)

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], **kwargs) -> List[Any]:
        return self.model.generate_batch(batch_prompts, **kwargs)

    def __repr__(self):
        return f"LLMModel(provider={self.provider}, model_name={self.model_name}, kwargs={self.model.kwargs})"

"""
OpenRouter Model Wrapper
"""

from typing import Any, Dict, List, Optional, Union
import os
import json
import logging
import copy
import base64
import io

from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)


class OpenRouterModel:
    """
    OpenRouter model wrapper that provides the same interface as OpenAIModel.
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenRouter model wrapper.

        Args:
            model_name: Name of the model to use (e.g., "openai/gpt-4o-mini")
            max_retries: Maximum number of retry attempts
            **kwargs: Additional arguments
        """
        api_key = os.environ["OPENROUTER_API_KEY"]

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        self.model_name = model_name
        self.max_retries = max(1, max_retries)
        self.kwargs = kwargs

    @staticmethod
    def _extract_json_block(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        return text

    def _parse_structured_response(self, response_text: str, text_format: type) -> Any:
        cleaned = self._extract_json_block(response_text)
        json_data = json.loads(cleaned)

        if hasattr(text_format, "model_validate"):
            return text_format.model_validate(json_data)
        if hasattr(text_format, "parse_obj"):
            return text_format.parse_obj(json_data)

        raise ValueError("text_format must be a Pydantic model class")

    @staticmethod
    def _sanitize_api_kwargs(api_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove local-only kwargs that are not valid OpenRouter Chat Completions params.
        """
        blocked_keys = {
            "fps",
            "nframes",
            "max_size",
            "max_size_video",
            "quality",
            "cache_dir",
        }
        dropped = blocked_keys.intersection(api_kwargs.keys())
        if dropped:
            logger.warning(
                "Dropping unsupported OpenRouter kwargs: %s",
                ", ".join(sorted(dropped)),
            )
        return {k: v for k, v in api_kwargs.items() if k not in blocked_keys}

    @staticmethod
    def _encode_image_to_data_url(image: Any) -> str:
        """Convert a PIL image (or image path) to a JPEG data URL for OpenRouter."""
        pil_image: Optional[Image.Image] = None
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            pil_image = Image.open(image)
        else:
            raise ValueError(f"Unsupported image payload type: {type(image)}")

        buffer = io.BytesIO()
        rgb_image = pil_image.convert("RGB")
        rgb_image.save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _preprocess_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize multimodal content to OpenRouter-compatible chat content format.
        """
        normalized = copy.deepcopy(messages)
        for message in normalized:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            new_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)  # type: ignore[arg-type]
                    continue

                item_type = item.get("type")
                if item_type == "text" and "text" in item:
                    new_content.append({"type": "text", "text": item["text"]})
                elif item_type == "image" and "image" in item:
                    image_url = self._encode_image_to_data_url(item["image"])
                    new_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
                elif item_type == "image_url" and "image_url" in item:
                    new_content.append(item)
                else:
                    new_content.append(item)

            message["content"] = new_content

        return normalized

    def generate(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs) -> Any:
        """
        Generate completion for a single prompt.

        Args:
            prompt: Conversation prompt (list of role/content dicts or raw string)

        Returns:
            Generated response string
        """
        # Normalize prompt to messages format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        messages = self._preprocess_messages(messages)

        text_format = kwargs.pop("text_format", None)

        merged_kwargs = self._sanitize_api_kwargs({**self.kwargs, **kwargs})

        # Move non-standard OpenAI params to extra_body for OpenRouter
        _extra_body_keys = {"top_k", "min_p", "repetition_penalty"}
        extra_body = {k: merged_kwargs.pop(k) for k in _extra_body_keys if k in merged_kwargs}
        if extra_body:
            merged_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **merged_kwargs,
        )

        if not response.choices:
            logger.warning("OpenRouter returned empty choices (model may not support this input type)")
            return None

        content = response.choices[0].message.content
        if text_format is None:
            return content

        try:
            return self._parse_structured_response(content, text_format)
        except Exception as e:
            logger.warning(f"Failed to parse structured response with {text_format}: {e}")
            raise

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], **kwargs) -> List[Any]:
        """
        Process multiple prompts in batch.

        Args:
            batch_prompts: List of conversation prompts

        Returns:
            List of response strings
        """
        results = []
        for prompt in batch_prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

import os
from pathlib import Path
import re
import time
from typing import Optional
import yaml

from worldmm.llm import LLMModel

model = LLMModel(model_name=os.getenv("TRANSLATION_MODEL", "openai/gpt-oss-120b"))


def _is_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    auth_markers = [
        "invalid api key",
        "incorrect api key",
        "authentication",
        "unauthorized",
        "401",
        "forbidden",
        "403",
        "api key",
    ]
    return any(marker in message for marker in auth_markers)


def call_gpt(
    prompt: str,
    system_message: str = "You are an effective first perspective assistant.",
    temperature=0.9,
    top_p=0.95,
    max_tokens=2200,
) -> Optional[str]:
    """
    Call GPT-4 API with given prompt and system message.
    Will automatically use standard OpenAI API using OPENAI_API_KEY.

    Args:
        prompt (str): The user prompt to send to GPT-4
        system_message (str): System message to set context for GPT-4
        temperature (float): Temperature parameter for response generation (deprecated)
        top_p (float): Top p parameter for response generation (deprecated)
        max_tokens (int): Maximum number of tokens in the response

    Returns:
        Optional[str]: The response content from GPT-4, or None if request fails after retries
    """
    provider = getattr(model, "provider", "openai")

    generation_kwargs = {}
    if provider == "openrouter":
        generation_kwargs["max_tokens"] = max_tokens * 2
    else:
        generation_kwargs["max_output_tokens"] = max_tokens * 2

    try:
        response = model.generate(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            **generation_kwargs
        )
        return response
    except Exception as exc:
        if provider == "openrouter" and _is_auth_error(exc) and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OpenRouter credentials are not properly configured. Set OPENROUTER_API_KEY."
            ) from exc

        if provider == "openai" and _is_auth_error(exc) and not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Standard OpenAI credentials are not properly configured (Azure OpenAI not supported)."
            ) from exc

        raise

def time_to_frame_idx(time_int: int, fps: int) -> int:
    """
    Convert time in HHMMSSFF format (integer or string) to frame index.
    :param time_int: Time in HHMMSSFF format, e.g., 10483000 (10:48:30.00) or "10483000".
    :param fps: Frames per second of the video.
    :return: Frame index corresponding to the given time.
    """
    # Ensure time_int is a string for slicing
    time_str = str(time_int).zfill(
        8
    )  # Pad with zeros if necessary to ensure it's 8 digits

    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    frames = int(time_str[6:8])

    total_seconds = hours * 3600 + minutes * 60 + seconds
    total_frames = total_seconds * fps + frames  # Convert to total frames

    return total_frames


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_video_paths(base_path, name):
    video_dir = Path(base_path) / name
    return [str(video_dir / video) for video in os.listdir(video_dir)]

def parse_evidence_output(response):
    """
    Parse the LLM output and return a dictionary with the appropriate format.

    Parameters:
    - response (str): The response from the LLM.

    Returns:
    - dict: A dictionary with the extracted status and information (if applicable).
    """
    if "I can't provide evidence." in response:
        return {"status": False}

    match = re.search(r"I can provide evidence\. Evidence: (.+)", response)
    if match:
        return {"status": True, "information": match.group(1).strip()}

    return {"status": False}

def extract_single_option_answer(answer):
    """
    Extracts the single option (A, B, C, or D) from the given answer string.

    Args:
        answer (str): The answer string containing the selected option.

    Returns:
        str: The selected option (A, B, C, or D).
    """
    match = re.search(r"\b([A-D])\b", answer)
    if match:
        return match.group(1)
    return None
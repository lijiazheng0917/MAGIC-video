from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "auto"):
        self.model_name = model_name
        self.device = device

        self.model = self._build_model()

    def _build_model(self) -> SentenceTransformer:
        # Prefer flash-attn when it is usable; otherwise fall back to SDPA/eager.
        attn_candidates = ["flash_attention_2", "sdpa", "eager"]
        last_error = None

        for attn_impl in attn_candidates:
            try:
                model = SentenceTransformer(
                    self.model_name,
                    model_kwargs={
                        "attn_implementation": attn_impl,
                        "dtype": "auto",
                        "device_map": self.device,
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
                # Truncate to 512 tokens to avoid OOM on very long texts (e.g. 1h captions).
                # Standard practice for dense retrieval; beginning of text is most informative.
                model.max_seq_length = 512
                return model
            except Exception as e:
                last_error = e

        raise RuntimeError(f"Failed to initialize {self.model_name} with any attention backend") from last_error
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, batch_size=batch_size)
        return embeddings
    
    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)

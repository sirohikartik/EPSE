"""CPU-optimized protein sequence embedder using ESM-2 or mock representations.

Designed specifically for CPU execution on resource-constrained systems,
enforcing torch CPU device, thread controls, batching, and L2 normalization.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Sequence, Union, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DEFAULT_EMBEDDING_DIM = 480


class ProteinEmbedder:
    """Generates dense vector representations for protein sequences on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_threads: Optional[int] = None,
        normalize: bool = True,
        use_mock: bool = False,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        cache_size: int = 1024,
        quantize: bool = False,
    ):
        """Initializes the CPU protein embedder.

        Args:
            model_name: HuggingFace model identifier for ESM-2.
            num_threads: Number of PyTorch CPU threads (defaults to torch default).
            normalize: Whether to L2-normalize embeddings (ideal for cosine similarity).
            use_mock: If True, uses deterministic hash/n-gram embeddings for testing without downloading weights.
            embedding_dim: Embedding dimension (480 for esm2_t12_35M).
            cache_size: Maximum number of embeddings to keep in LRU cache (0 to disable).
            quantize: If True, applies dynamic int8 quantization on Linear layers for accelerated CPU inference.
        """
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.normalize = normalize
        self.use_mock = use_mock
        self.embedding_dim = embedding_dim
        self.cache_size = max(0, int(cache_size))
        self.quantize = quantize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._model = None
        self._tokenizer = None

        if num_threads is not None and num_threads > 0:
            torch.set_num_threads(num_threads)

        if not self.use_mock:
            self._init_model()

    def _init_model(self) -> None:
        """Loads ESM-2 model and tokenizer onto CPU."""
        logger.info("Initializing ESM-2 model (%s) on CPU...", self.model_name)
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Update dimension from model configuration if available
        if hasattr(self._model.config, "hidden_size"):
            self.embedding_dim = self._model.config.hidden_size

        if self.quantize:
            try:
                if (
                    hasattr(torch.backends, "quantized")
                    and "qnnpack" in torch.backends.quantized.supported_engines
                ):
                    torch.backends.quantized.engine = "qnnpack"
                logger.info("Applying dynamic int8 quantization to model linear layers on CPU...")
                self._model = torch.ao.quantization.quantize_dynamic(
                    self._model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                logger.warning("Dynamic quantization failed; proceeding with fp32: %s", e)

    def _mock_embed(self, sequences: Sequence[str]) -> np.ndarray:
        """Generates deterministic pseudo-embeddings for testing without heavy model weights."""
        rng = np.random.RandomState(42)
        embeddings = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)

        for i, seq in enumerate(sequences):
            # Seed based on sequence content for determinism
            seq_seed = sum(ord(c) * (j + 1) for j, c in enumerate(seq[:64])) % (2**31 - 1)
            local_rng = np.random.RandomState(seq_seed)
            vec = local_rng.randn(self.embedding_dim).astype(np.float32)
            if self.normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings[i] = vec

        return embeddings

    def _cache_get(self, seq: str) -> Optional[np.ndarray]:
        """Retrieves cached embedding if present, updating LRU order."""
        if self.cache_size <= 0 or seq not in self._cache:
            return None
        self._cache.move_to_end(seq)
        return self._cache[seq]

    def _cache_put(self, seq: str, vec: np.ndarray) -> None:
        """Stores embedding in LRU cache, evicting oldest if full."""
        if self.cache_size <= 0:
            return
        if seq in self._cache:
            self._cache.move_to_end(seq)
        else:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[seq] = vec

    def clear_cache(self) -> None:
        """Clears the embedding LRU cache."""
        self._cache.clear()

    @property
    def cache_size_current(self) -> int:
        """Returns the number of currently cached embeddings."""
        return len(self._cache)

    def _compute_embeddings(
        self,
        sequences: Sequence[str],
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> np.ndarray:
        """Internal helper to compute embeddings via PyTorch on CPU."""
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(sequences), batch_size):
            batch = list(sequences[i : i + batch_size])
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                last_hidden = outputs.last_hidden_state  # shape: (B, seq_len, D)

                mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                if self.normalize:
                    mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

                batch_embeddings = mean_pooled.cpu().numpy().astype(np.float32)
                all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def embed_sequences(
        self,
        sequences: Sequence[str],
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> np.ndarray:
        """Computes embeddings for a list of protein sequences on CPU.

        Utilizes the internal LRU cache to avoid redundant computation for previously
        embedded queries.

        Args:
            sequences: List or sequence of amino acid strings.
            batch_size: Number of sequences to process concurrently in a CPU batch.
            max_length: Maximum sequence token length for truncation.

        Returns:
            np.ndarray of shape (len(sequences), embedding_dim) in float32.
        """
        if not sequences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if self.cache_size > 0:
            cached_hits: dict[int, np.ndarray] = {}
            miss_indices: list[int] = []
            miss_seqs: list[str] = []

            for idx, seq in enumerate(sequences):
                cached = self._cache_get(seq)
                if cached is not None:
                    cached_hits[idx] = cached
                else:
                    miss_indices.append(idx)
                    miss_seqs.append(seq)

            if not miss_seqs:
                return np.vstack([cached_hits[i] for i in range(len(sequences))])

            if self.use_mock or self._model is None:
                computed = self._mock_embed(miss_seqs)
            else:
                computed = self._compute_embeddings(miss_seqs, batch_size=batch_size, max_length=max_length)

            for seq, vec in zip(miss_seqs, computed):
                self._cache_put(seq, vec)

            all_res = np.empty((len(sequences), self.embedding_dim), dtype=np.float32)
            for idx, hit in cached_hits.items():
                all_res[idx] = hit
            for local_i, orig_i in enumerate(miss_indices):
                all_res[orig_i] = computed[local_i]
            return all_res

        if self.use_mock or self._model is None:
            return self._mock_embed(sequences)

        return self._compute_embeddings(sequences, batch_size=batch_size, max_length=max_length)

    def embed_single(self, sequence: str) -> np.ndarray:
        """Computes embedding for a single protein sequence, with LRU cache lookup."""
        cached = self._cache_get(sequence)
        if cached is not None:
            return cached.copy()
        vec = self.embed_sequences([sequence], batch_size=1)[0]
        return vec.copy()

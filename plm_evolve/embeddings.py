"""PLM-agnostic embedding interface plus an ESM-2 implementation.

The rest of the package depends only on the :class:`EmbeddingExtractor`
contract: given a FASTA, return a DataFrame indexed by sequence id whose
columns are embedding dimensions. ESM-2 is the single bundled backend, but
new protein language models can be added by implementing the same protocol.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO

from .variants import WT_TOKEN, parse_variant

DEFAULT_ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

# ESM-2 was trained with a 1022-residue context window (excluding special
# tokens). Sequences longer than this are truncated with a warning.
DEFAULT_MAX_RESIDUES = 1022

# ESM-C default backbone. The ``esm`` SDK exposes "esmc_300m" (width 960) and
# "esmc_600m" (width 1152). ESM-C uses a large position budget; this default is
# deliberately generous and is NOT load-bearing (truncation is manual and the
# tests monkeypatch _forward). Override via max_residues if a tighter window is
# wanted for a specific model.
DEFAULT_ESMC_MODEL = "esmc_300m"
DEFAULT_ESMC_MAX_RESIDUES = 2046

# Expected per-residue embedding width per ESM-C backbone, used only for a
# runtime self-check inside the real _forward. NEVER branch on these values.
ESMC_EXPECTED_WIDTH = {"esmc_300m": 960, "esmc_600m": 1152}

# Supported pooling modes. 'mean' is the default and stays byte-identical to
# the original behavior; 'delta' and 'mutated_token' are OPT-IN GAP-2 modes.
POOLING_MODES = ("mean", "delta", "mutated_token")


class EmbeddingExtractor(ABC):
    """Abstract base for any PLM embedding backend.

    Implementations must turn a FASTA file into a per-sequence embedding
    matrix. The pooling convention is a mean over residue positions only
    (special tokens excluded).
    """

    @abstractmethod
    def embed_fasta(self, fasta_path: Union[str, Path]) -> pd.DataFrame:
        """Embed every record in ``fasta_path``.

        Returns a DataFrame indexed by sequence id, columns = embedding dims.
        """
        raise NotImplementedError


def read_fasta(fasta_path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Read a FASTA into a list of ``(id, sequence)`` tuples."""
    records = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        records.append((record.id, str(record.seq)))
    if not records:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")
    return records


def masked_mean_pool(
    hidden_state: "np.ndarray",
    attention_mask: "np.ndarray",
    special_tokens_mask: "np.ndarray",
) -> np.ndarray:
    """Mean-pool residue embeddings, excluding padding and special tokens.

    The ESM-2 tokenizer prepends ``<cls>`` and appends ``<eos>``; both carry
    ``attention_mask == 1``. Pooling over the attention mask alone would mix
    those special tokens into the per-protein vector. Here the pooling weight
    is ``attention_mask * (1 - special_tokens_mask)`` so only true residues
    contribute, matching the ESM/EVOLVEpro mean-representation convention.

    Args:
        hidden_state: ``(batch, seq_len, dim)`` last hidden state.
        attention_mask: ``(batch, seq_len)`` 1 for real tokens, 0 for padding.
        special_tokens_mask: ``(batch, seq_len)`` 1 for special tokens.

    Returns:
        ``(batch, dim)`` mean-pooled embeddings.
    """
    hidden_state = np.asarray(hidden_state, dtype=np.float64)
    attention_mask = np.asarray(attention_mask, dtype=np.float64)
    special_tokens_mask = np.asarray(special_tokens_mask, dtype=np.float64)

    residue_mask = attention_mask * (1.0 - special_tokens_mask)  # (batch, seq)
    denom = residue_mask.sum(axis=1, keepdims=True)  # (batch, 1)
    if np.any(denom == 0):
        raise ValueError(
            "A sequence has zero residue tokens after masking special tokens; "
            "cannot mean-pool."
        )
    weighted = (hidden_state * residue_mask[:, :, None]).sum(axis=1)  # (batch,dim)
    return weighted / denom


def residue_count_from_masks(
    attention_mask: "np.ndarray",
    special_tokens_mask: "np.ndarray",
) -> np.ndarray:
    """Per-row count of true residue tokens (padding and special tokens excluded).

    ``attention_mask * (1 - special_tokens_mask)`` is 1 only for real residue
    tokens, so its row-sum equals the post-truncation residue count L'. This is
    the authoritative bound for ``mutated_token`` pooling: because <eos> and
    padding are excluded, a 1-based position p satisfying ``p <= L'`` is
    guaranteed to index a real residue (and not the <eos> / a padding slot).

    Args:
        attention_mask: ``(batch, seq_len)`` 1 for real tokens, 0 for padding.
        special_tokens_mask: ``(batch, seq_len)`` 1 for special tokens.

    Returns:
        ``(batch,)`` integer residue counts.
    """
    attention_mask = np.asarray(attention_mask, dtype=np.float64)
    special_tokens_mask = np.asarray(special_tokens_mask, dtype=np.float64)
    residue_mask = attention_mask * (1.0 - special_tokens_mask)
    return residue_mask.sum(axis=1).astype(np.int64)


def mutated_token_pool(
    hidden_state: "np.ndarray",
    position_token_index: int,
) -> np.ndarray:
    """Return the per-residue hidden state AT a single mutated token index.

    The ESM-2 tokenizer prepends ``<cls>`` at token index 0, so the 1-based
    residue position p maps directly to token index p (residue 1 -> index 1).
    This identity holds under right-padding (the transformers/ESM default):
    the <cls> token is always at index 0 and residues follow contiguously.

    Args:
        hidden_state: ``(seq_len, dim)`` last hidden state for ONE sequence.
        position_token_index: token index of the mutated residue (== 1-based
            position). The caller is responsible for validating it against the
            post-truncation residue count via :func:`residue_count_from_masks`.

    Returns:
        ``(dim,)`` hidden state at that token.
    """
    hidden_state = np.asarray(hidden_state, dtype=np.float64)
    return hidden_state[position_token_index]


class _PoolingMixin(EmbeddingExtractor):
    """Backend-agnostic pooling + FASTA dispatch shared by all extractors.

    A concrete backend supplies only three things:

    - ``self.pooling`` / ``self.batch_size`` / ``self.max_residues`` attributes
      (set in its ``__init__``);
    - ``_load(self) -> None`` to lazily build the model;
    - ``_forward(self, sequences) -> (hidden_state, attention_mask,
      special_tokens_mask)`` returning numpy arrays in the SAME convention as
      :meth:`ESM2Extractor._forward` (special token index 0 = <cls>/BOS, the
      trailing <eos>/EOS also flagged, right-padding). Because the masks encode
      where the residues are, every pooling mode below transfers unchanged to a
      new backend: masked mean, delta (WT-relative), and mutated-token all
      reuse the pure helpers and the same WT / multi-mutant / window guards.
    """

    def embed_fasta(self, fasta_path: Union[str, Path]) -> pd.DataFrame:
        records = read_fasta(fasta_path)
        self._load()

        if self.pooling == "mean":
            ids, vectors = self._embed_mean(records)
        elif self.pooling == "delta":
            ids, vectors = self._embed_delta(records)
        elif self.pooling == "mutated_token":
            ids, vectors = self._embed_mutated_token(records)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unknown pooling mode {self.pooling!r}.")

        matrix = np.vstack(vectors)
        df = pd.DataFrame(matrix, index=ids)
        df.index.name = "variant"
        return df

    # --- per-mode pooling (backend-agnostic; uses _forward + pure helpers) ---

    def _embed_mean(
        self, records: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Streaming masked-mean path over residue tokens only."""
        ids: List[str] = []
        vectors: List[np.ndarray] = []

        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            batch_ids = [rid for rid, _ in batch]
            batch_seqs = [seq for _, seq in batch]

            hidden_state, attention_mask, special_tokens_mask = self._forward(
                batch_seqs
            )
            pooled = masked_mean_pool(
                hidden_state, attention_mask, special_tokens_mask
            )

            ids.extend(batch_ids)
            vectors.extend(pooled)

        return ids, vectors

    def _embed_delta(
        self, records: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Per-variant mean minus the WT mean (constant-offset; see ESM2 docstring)."""
        ids, vectors = self._embed_mean(records)
        id_to_idx = {rid: i for i, rid in enumerate(ids)}
        if WT_TOKEN not in id_to_idx:
            raise ValueError(
                f"pooling='delta' requires a record with id {WT_TOKEN!r} in the "
                f"FASTA (its mean-pool is the subtracted reference), but none "
                f"was found. Found ids include: {ids[:5]!r}..."
            )
        wt_vec = vectors[id_to_idx[WT_TOKEN]]
        out: List[np.ndarray] = []
        for rid, vec in zip(ids, vectors):
            if rid == WT_TOKEN:
                out.append(np.zeros_like(wt_vec))  # WT's own delta is exactly 0.
            else:
                out.append(vec - wt_vec)
        return ids, out

    def _embed_mutated_token(
        self, records: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Per-variant hidden state at the mutated token (GAP-2; see ESM2 docstring).

        The WT record has no mutated position, so its row is a zero vector
        (documented). Multi-mutant ids fail fast via ``parse_variant``. A
        mutated position past the post-truncation residue window raises a clear
        error rather than silently falling back.
        """
        ids: List[str] = []
        vectors: List[np.ndarray] = []

        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            batch_ids = [rid for rid, _ in batch]
            batch_seqs = [seq for _, seq in batch]

            hidden_state, attention_mask, special_tokens_mask = self._forward(
                batch_seqs
            )
            residue_counts = residue_count_from_masks(
                attention_mask, special_tokens_mask
            )
            dim = hidden_state.shape[-1]

            for b, rid in enumerate(batch_ids):
                if rid == WT_TOKEN:
                    # WT has no mutated position -> zero vector (documented).
                    vectors.append(np.zeros(dim, dtype=np.float64))
                    ids.append(rid)
                    continue

                mut = parse_variant(rid)  # raises on multi-mutant / malformed.
                # <cls>/BOS is token index 0, so 1-based residue position ==
                # token index (right-padding assumption; see mutated_token_pool).
                token_index = mut.position
                residue_count = int(residue_counts[b])
                if mut.position < 1 or mut.position > residue_count:
                    raise ValueError(
                        f"Variant {rid!r} mutated position {mut.position} is "
                        f"outside the post-truncation residue window "
                        f"(L'={residue_count}, max_residues={self.max_residues}); "
                        f"the mutated-token embedding is undefined. Refusing to "
                        f"silently fall back."
                    )
                vectors.append(
                    mutated_token_pool(hidden_state[b], token_index)
                )
                ids.append(rid)

        return ids, vectors


class ESM2Extractor(_PoolingMixin):
    """ESM-2 embedding backend via HuggingFace transformers.

    Loads ``facebook/esm2_t33_650M_UR50D`` by default, uses fp16 on GPU,
    batches sequences, and pools according to ``pooling`` (default ``'mean'``).

    Pooling modes (``pooling``):

    - ``'mean'`` (default): masked mean over residue positions only. This is
      the original behavior and stays byte-identical: for any given input it
      produces the same numbers as before this option existed.
    - ``'delta'``: per-variant (mean-pool of the variant) MINUS (mean-pool of
      the ``WT`` record). Requires a record with id ``WT`` in the FASTA;
      missing WT fails fast. The WT row itself is the zero vector. NOTE: under
      a RandomForest top-layer, ``'delta'`` is PROVABLY EQUIVALENT to
      ``'mean'`` -- subtracting a constant vector (the WT mean) from every row
      shifts all candidate split thresholds by the same constant, leaving the
      tree structure and therefore every prediction identical. It exists as a
      CANARY (greedy-on-delta must equal greedy-on-mean) and for non-tree
      regressors where the offset is not absorbed.
    - ``'mutated_token'``: the per-residue hidden state AT the mutated
      position (GAP-2). The mutated position is parsed from each variant id via
      :func:`plm_evolve.variants.parse_variant`; the WT record (no mutated
      position) gets a zero vector. Unlike ``'delta'``, this changes the
      information content reaching the top-layer: it preserves the changed
      position marginal instead of diluting it by ~1/L. Whether it OUTPERFORMS
      mean is NOT asserted here: mutated-token wins IFF the variant-effect
      signal is local (high locality alpha) -- an empirical property to be
      measured on real embeddings, never hardcoded or claimed a priori.

    The model and forward pass are split into small methods so tests can
    monkeypatch loading/forward without downloading weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM2_MODEL,
        batch_size: int = 8,
        max_residues: int = DEFAULT_MAX_RESIDUES,
        device: str = None,
        pooling: str = "mean",
    ) -> None:
        if pooling not in POOLING_MODES:
            raise ValueError(
                f"Unknown pooling mode {pooling!r}; expected one of "
                f"{POOLING_MODES}."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_residues = max_residues
        self._device = device
        self.pooling = pooling
        self._tokenizer = None
        self._model = None

    # --- load + forward split out for testability -------------------------

    def _load(self) -> None:
        """Lazily load the tokenizer and model (kept tiny for monkeypatching)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        if self._device == "cuda":
            model = model.half()
        model = model.to(self._device)
        model.eval()
        self._model = model

    def _forward(
        self, sequences: List[str]
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Run one batch forward pass.

        Returns ``(hidden_state, attention_mask, special_tokens_mask)`` as
        numpy arrays. Truncates sequences longer than ``max_residues`` and
        warns on stderr.
        """
        import torch

        # +2 for the <cls> and <eos> special tokens the tokenizer adds.
        max_length = self.max_residues + 2

        for seq in sequences:
            if len(seq) > self.max_residues:
                print(
                    f"[plm-evolve] WARNING: sequence of length {len(seq)} "
                    f"exceeds max_residues={self.max_residues}; truncating.",
                    file=sys.stderr,
                )

        encoded = self._tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        inputs = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        hidden_state = outputs.last_hidden_state.float().cpu().numpy()
        attention_mask = encoded["attention_mask"].cpu().numpy()
        return hidden_state, attention_mask, special_tokens_mask.cpu().numpy()


class ESMCExtractor(_PoolingMixin):
    """ESM-C embedding backend via the ``esm`` SDK.

    Drop-in interchangeable with :class:`ESM2Extractor`: same constructor shape,
    same ``embed_fasta`` DataFrame contract (index name ``"variant"``, one row
    per record incl WT), and the SAME three pooling modes with identical
    semantics and WT / multi-mutant / window guards. Interchangeability is
    achieved by making ``_forward`` return the EXACT same
    ``(hidden_state, attention_mask, special_tokens_mask)`` numpy tuple
    convention as ESM-2 (BOS at token index 0, EOS after the last residue,
    right-padding over the batch), so all of :class:`_PoolingMixin` transfers
    unchanged.

    ESM-C residue embeddings come from ``logits(..., return_embeddings=True)``:
    ``o.embeddings[0]`` has shape ``(L+2, width)`` with index 0 = BOS and
    index -1 = EOS, so residue 1 lands at token index 1 -- the same 1-based
    position == token index convention used by ESM-2 ``mutated_token`` pooling.

    The SDK ``encode`` is per-protein, so the batch is processed one sequence at
    a time and right-padded into the batch tensor. ``_load`` and ``_forward``
    are split out so tests can monkeypatch them WITHOUT downloading weights.

    Pooling modes (``pooling``) are identical to :class:`ESM2Extractor`; see its
    docstring for the per-mode semantics (mean / delta / mutated_token).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESMC_MODEL,
        batch_size: int = 8,
        max_residues: int = DEFAULT_ESMC_MAX_RESIDUES,
        device: str = None,
        pooling: str = "mean",
    ) -> None:
        if pooling not in POOLING_MODES:
            raise ValueError(
                f"Unknown pooling mode {pooling!r}; expected one of "
                f"{POOLING_MODES}."
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_residues = max_residues
        self._device = device
        self.pooling = pooling
        self._model = None

    # --- load + forward split out for testability -------------------------

    def _load(self) -> None:
        """Lazily import esm and build the ESM-C model (kept tiny for monkeypatching)."""
        if self._model is not None:
            return
        import torch
        from esm.models.esmc import ESMC

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model = ESMC.from_pretrained(self.model_name).to(self._device).eval()
        self._model = model

    def _encode_one(self, seq: str) -> "np.ndarray":
        """Embed ONE sequence -> ``(L+2, width)`` numpy array (BOS .. residues .. EOS).

        Truncates to ``max_residues`` (the SDK does not truncate) with the same
        stderr warning as ESM-2, then runs encode + logits. A runtime width
        self-check warns on an unexpected width but NEVER branches on it.
        """
        import torch
        from esm.sdk.api import ESMProtein, LogitsConfig

        if len(seq) > self.max_residues:
            print(
                f"[plm-evolve] WARNING: sequence of length {len(seq)} "
                f"exceeds max_residues={self.max_residues}; truncating.",
                file=sys.stderr,
            )
            seq = seq[: self.max_residues]

        with torch.no_grad():
            tensor = self._model.encode(ESMProtein(sequence=seq))
            out = self._model.logits(
                tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
        # out.embeddings: (1, L+2, width) -> first (and only) protein.
        emb = out.embeddings[0].float().cpu().numpy()

        expected = ESMC_EXPECTED_WIDTH.get(self.model_name)
        if expected is not None and emb.shape[-1] != expected:
            # Self-check only: warn, do not branch on / enforce the width.
            print(
                f"[plm-evolve] WARNING: ESM-C width {emb.shape[-1]} != expected "
                f"{expected} for model {self.model_name!r}.",
                file=sys.stderr,
            )
        return emb

    def _forward(
        self, sequences: List[str]
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Embed a batch one sequence at a time and right-pad to the ESM-2 tuple.

        Returns ``(hidden_state, attention_mask, special_tokens_mask)`` numpy
        arrays in the SAME convention as :meth:`ESM2Extractor._forward`: token
        index 0 = BOS special, the trailing EOS special, residues in between,
        right-padding to the batch max length.
        """
        per_seq = [self._encode_one(seq) for seq in sequences]  # each (Li+2, W)
        width = per_seq[0].shape[-1]
        # token length = residues + BOS + EOS; per-sequence residue count is
        # rows-2 (BOS + EOS), which already reflects any truncation above.
        token_lens = [emb.shape[0] for emb in per_seq]
        max_len = max(token_lens)
        batch = len(sequences)

        hidden = np.zeros((batch, max_len, width), dtype=np.float64)
        attention = np.zeros((batch, max_len), dtype=np.float64)
        special = np.zeros((batch, max_len), dtype=np.float64)

        for b, emb in enumerate(per_seq):
            tlen = emb.shape[0]  # = residues + 2
            hidden[b, :tlen] = emb
            attention[b, :tlen] = 1.0
            special[b, 0] = 1.0  # BOS
            special[b, tlen - 1] = 1.0  # EOS
        return hidden, attention, special


def make_extractor(
    model_name: str, pooling: str = "mean", **kw
) -> EmbeddingExtractor:
    """Backend factory: pick ESM-C vs ESM-2 from the model tag.

    ``model_name`` starting with ``"esmc"`` (e.g. ``"esmc_300m"`` /
    ``"esmc_600m"``) -> :class:`ESMCExtractor`; anything else (``"facebook/esm2*"``
    or an arbitrary HF id) -> :class:`ESM2Extractor`. Extra keyword arguments
    (``batch_size``, ``max_residues``, ``device``) pass through to the backend.
    Dispatch is import-free and does NOT load weights (``_load`` stays lazy).
    """
    if model_name.startswith("esmc"):
        return ESMCExtractor(model_name=model_name, pooling=pooling, **kw)
    return ESM2Extractor(model_name=model_name, pooling=pooling, **kw)


def to_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write an embedding DataFrame to CSV (variant id as the index)."""
    df.to_csv(str(path), index=True)


def from_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read an embedding DataFrame from CSV (first column is the variant id)."""
    df = pd.read_csv(str(path), index_col=0)
    df.index.name = "variant"
    return df

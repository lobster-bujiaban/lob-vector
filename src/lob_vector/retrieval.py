"""BM25、RRF 融合、轻量重排与检索指标。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .models import Chunk, SearchResult

_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """生成中英文 token，并为中文保留相邻二元组。"""
    base = _TOKEN_PATTERN.findall(text.casefold())
    return [*base, *(f"{a}{b}" for a, b in zip(base, base[1:], strict=False))]


@dataclass(slots=True)
class BM25Retriever:
    chunks: Sequence[Chunk]
    k1: float = 1.5
    b: float = 0.75
    _tokens: list[list[str]] = field(init=False, repr=False)
    _frequencies: list[Counter[str]] = field(init=False, repr=False)
    _document_frequency: Counter[str] = field(init=False, repr=False)
    _average_length: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = [tokenize(chunk.content) for chunk in self.chunks]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )
        self._average_length = (
            sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        )

    def search(self, query: str, *, top_k: int = 10) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query_tokens = set(tokenize(query))
        scored = []
        total = len(self.chunks)
        for chunk, tokens, frequencies in zip(
            self.chunks, self._tokens, self._frequencies, strict=True
        ):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_frequency = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_factor = frequency + self.k1 * (
                    1
                    - self.b
                    + self.b * len(tokens) / max(self._average_length, 1.0)
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / length_factor
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], *, top_k: int = 10, k: int = 60
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for result in ranking:
            chunks[result.chunk.id] = result.chunk
            scores[result.chunk.id] = scores.get(result.chunk.id, 0.0) + 1 / (
                k + result.rank
            )
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        SearchResult(chunk=chunks[chunk_id], score=scores[chunk_id], rank=rank)
        for rank, chunk_id in enumerate(ordered[:top_k], start=1)
    ]


def rerank(query: str, results: Sequence[SearchResult], *, top_k: int = 10) -> list[SearchResult]:
    """使用词覆盖、章节命中与原排名进行透明的本地重排。"""
    query_tokens = set(tokenize(query))
    rescored = []
    for result in results:
        content_tokens = set(tokenize(result.chunk.content))
        section_tokens = set(tokenize(str(result.chunk.metadata.get("section", ""))))
        coverage = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        section_coverage = len(query_tokens & section_tokens) / max(len(query_tokens), 1)
        rank_score = 1 / result.rank
        score = 0.65 * coverage + 0.2 * section_coverage + 0.15 * rank_score
        rescored.append((score, result.chunk))
    rescored.sort(key=lambda item: (-item[0], item[1].id))
    return [
        SearchResult(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(rescored[:top_k], start=1)
    ]


def ranking_metrics(
    rankings: Sequence[Sequence[SearchResult]],
    expected: Sequence[tuple[str, str]],
    *,
    top_k: int = 3,
) -> dict[str, float]:
    hits = []
    reciprocal_ranks = []
    discounted_gains = []
    for ranking, (source_name, section_name) in zip(rankings, expected, strict=True):
        rank = next(
            (
                result.rank
                for result in ranking[:top_k]
                if str(result.chunk.metadata.get("source", "")).endswith(source_name)
                and section_name in str(result.chunk.metadata.get("section", ""))
            ),
            0,
        )
        hits.append(1.0 if rank else 0.0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        discounted_gains.append(1 / math.log2(rank + 1) if rank else 0.0)
    count = max(len(rankings), 1)
    return {
        f"recall_at_{top_k}": sum(hits) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "ndcg": sum(discounted_gains) / count,
    }

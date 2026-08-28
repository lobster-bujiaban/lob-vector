"""统一向量存储抽象与内存实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

from .models import Chunk, MetadataValue, SearchResult

FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte"]


@dataclass(frozen=True, slots=True)
class MetadataCondition:
    """一个 Metadata 比较条件。"""

    key: str
    operator: FilterOperator
    value: MetadataValue

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("过滤字段不能为空")

    def matches(self, chunk: Chunk) -> bool:
        if self.key not in chunk.metadata:
            return False
        actual = chunk.metadata[self.key]
        if self.operator == "eq":
            return actual == self.value
        if self.operator == "ne":
            return actual != self.value
        if isinstance(actual, bool) or isinstance(self.value, bool):
            return False
        if not isinstance(actual, (int, float, str)) or not isinstance(
            self.value, type(actual)
        ):
            return False
        if self.operator == "gt":
            return actual > self.value
        if self.operator == "gte":
            return actual >= self.value
        if self.operator == "lt":
            return actual < self.value
        return actual <= self.value


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """按 AND 组合的一组 Metadata 条件。"""

    conditions: tuple[MetadataCondition, ...] = ()

    def matches(self, chunk: Chunk) -> bool:
        return all(condition.matches(chunk) for condition in self.conditions)


@runtime_checkable
class VectorStore(Protocol):
    """向量存储需要提供的最小能力。"""

    @property
    def dimension(self) -> int: ...

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]: ...


@dataclass(slots=True)
class MemoryVectorStore:
    """使用全量扫描和余弦相似度进行查询的内存存储。"""

    dimension: int
    _entries: dict[str, tuple[Chunk, tuple[float, ...]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension 必须是整数")
        if self.dimension <= 0:
            raise ValueError("dimension 必须大于 0")

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")
        prepared: list[tuple[Chunk, tuple[float, ...]]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            prepared.append((chunk, self._validate_vector(vector)))
        for chunk, vector in prepared:
            self._entries[chunk.id] = (chunk, vector)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        query = self._validate_vector(query_vector)
        scored = [
            (self._cosine_similarity(query, vector), chunk)
            for chunk, vector in self._entries.values()
            if metadata_filter is None or metadata_filter.matches(chunk)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def _validate_vector(self, vector: Sequence[float]) -> tuple[float, ...]:
        if len(vector) != self.dimension:
            raise ValueError(f"向量维度必须为 {self.dimension}，当前为 {len(vector)}")
        values = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("向量只能包含有限数值")
        if not any(values):
            raise ValueError("向量不能是零向量")
        return values

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm)

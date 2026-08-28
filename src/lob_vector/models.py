"""向量检索主链路使用的核心数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping, TypeAlias

MetadataValue: TypeAlias = str | int | float | bool | None
Metadata: TypeAlias = Mapping[str, MetadataValue]


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")


def _freeze_metadata(metadata: Metadata) -> Metadata:
    copied: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        _require_text(key, "metadata key")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(f"metadata[{key!r}] 不是支持的标量类型")
        copied[key] = value
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class Document:
    """一份进入检索系统的原始文本及其来源信息。"""

    id: str
    content: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.id, "document.id")
        _require_text(self.content, "document.content")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Chunk:
    """从 Document 切分出的可检索文本块。"""

    id: str
    document_id: str
    content: str
    index: int
    start: int
    end: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.id, "chunk.id")
        _require_text(self.document_id, "chunk.document_id")
        _require_text(self.content, "chunk.content")
        if self.index < 0:
            raise ValueError("chunk.index 不能小于 0")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("chunk 的字符范围必须满足 0 <= start < end")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一次向量查询返回的 Chunk、相似度和排名。"""

    chunk: Chunk
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isfinite(self.score):
            raise ValueError("search_result.score 必须是有限数值")
        if self.rank < 1:
            raise ValueError("search_result.rank 必须从 1 开始")

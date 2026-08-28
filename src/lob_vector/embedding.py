"""文本 Embedding 抽象与本地确定性实现。"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """将一批文本转换为同维度向量的统一接口。"""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class HashEmbedder:
    """使用稳定哈希生成归一化向量，仅用于本地学习和链路验证。"""

    dimension: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool):
            raise TypeError("dimension 必须是整数")
        if self.dimension <= 0:
            raise ValueError("dimension 必须大于 0")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding 文本不能为空")

        tokens = self._tokens(text)
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("文本无法生成有效向量")
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = text.casefold()
        tokens = _TOKEN_PATTERN.findall(normalized)
        if not tokens:
            return list(normalized.strip())

        # 中文单字之外再加入相邻 token，保留少量局部词序信息。
        tokens.extend(
            f"{left}{right}" for left, right in zip(tokens, tokens[1:], strict=False)
        )
        return tokens

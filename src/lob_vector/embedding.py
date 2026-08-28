"""文本 Embedding 抽象与本地确定性实现。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


@dataclass(frozen=True, slots=True)
class BailianEmbedder:
    """通过百炼 OpenAI 兼容接口生成真实语义向量。"""

    dimension: int = 1024
    model: str = "text-embedding-v4"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    batch_size: int = 10
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool):
            raise TypeError("dimension 必须是整数")
        if self.dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        if not self.model.strip():
            raise ValueError("百炼模型名称不能为空")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("百炼 base_url 必须是 HTTP(S) 地址")
        if not self.api_key_env.strip():
            raise ValueError("API Key 环境变量名不能为空")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.timeout <= 0:
            raise ValueError("timeout 必须大于 0")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in prepared):
            raise ValueError("Embedding 文本不能为空")

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"未设置百炼 API Key 环境变量：{self.api_key_env}")

        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            vectors.extend(self._request(prepared[start : start + self.batch_size], api_key))
        return vectors

    def _request(self, texts: list[str], api_key: str) -> list[list[float]]:
        payload = json.dumps(
            {
                "model": self.model,
                "input": texts,
                "dimensions": self.dimension,
                "encoding_format": "float",
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"百炼 Embedding 请求失败（HTTP {error.code}）：{detail}") from error
        except URLError as error:
            raise RuntimeError(f"无法连接百炼 Embedding 服务：{error.reason}") from error

        data = result.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("百炼 Embedding 响应中的向量数量不正确")
        ordered = sorted(data, key=lambda item: item.get("index", -1))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vector, list) or len(vector) != self.dimension for vector in vectors):
            raise RuntimeError(f"百炼 Embedding 响应向量维度不是 {self.dimension}")
        return [[float(value) for value in vector] for vector in vectors]

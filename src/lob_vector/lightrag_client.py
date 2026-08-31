"""LightRAG Server REST 客户端，用于产品对照实验。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class LightRAGClient:
    base_url: str = "http://127.0.0.1:9621"
    api_key_env: str = "LIGHTRAG_API_KEY"
    timeout: float = 180.0

    @classmethod
    def from_env(cls) -> "LightRAGClient":
        return cls(
            base_url=os.getenv("LIGHTRAG_URL", "http://127.0.0.1:9621").rstrip("/"),
            timeout=float(os.getenv("LIGHTRAG_TIMEOUT", "180")),
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def insert_texts(self, texts: list[str]) -> dict[str, Any]:
        if not texts:
            raise ValueError("没有可写入 LightRAG 的文档")
        return self._request("POST", "/documents/texts", {"texts": texts})

    def query(self, question: str, mode: str) -> dict[str, Any]:
        if mode not in {"naive", "local", "global", "hybrid", "mix"}:
            raise ValueError(f"不支持的 LightRAG 查询模式：{mode}")
        started = time.perf_counter()
        result = self._request(
            "POST",
            "/query",
            {
                "query": question,
                "mode": mode,
                "include_references": True,
                "include_chunk_content": True,
                "top_k": 10,
                "chunk_top_k": 5,
                "enable_rerank": False,
            },
        )
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"未设置 LightRAG 密钥：{self.api_key_env}")
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LightRAG 请求失败（HTTP {error.code}）：{detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"无法连接 LightRAG：{self.base_url}（{error.reason}）"
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError("LightRAG 返回格式不正确")
        return result

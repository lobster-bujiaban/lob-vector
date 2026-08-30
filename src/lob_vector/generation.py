"""基于检索证据的文本生成能力。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import SearchResult

REFUSAL_TEXT = "根据当前知识库资料，无法回答这个问题。"


@dataclass(frozen=True, slots=True)
class BailianChatGenerator:
    """通过百炼 OpenAI 兼容 Chat Completions 生成带引用回答。"""

    model: str = "qwen-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout: float = 60.0

    def generate(
        self, question: str, evidence: Sequence[SearchResult]
    ) -> tuple[str, dict[str, int]]:
        if not evidence:
            return REFUSAL_TEXT, {}
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"未设置百炼 API Key 环境变量：{self.api_key_env}")

        context = "\n\n".join(
            self._evidence_block(number, result)
            for number, result in enumerate(evidence, start=1)
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是严格基于证据回答的知识库助手。只能使用用户提供的证据，"
                            "禁止补充外部知识或猜测。每个事实结论后必须标注对应证据编号，"
                            "格式为 [1] 或 [1][2]。若证据不足以回答，必须只输出："
                            f"{REFUSAL_TEXT}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"问题：{question}\n\n证据：\n{context}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 800,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
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
            raise RuntimeError(
                f"百炼文本生成失败（HTTP {error.code}）：{detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"无法连接百炼文本生成服务：{error.reason}") from error

        try:
            answer = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise RuntimeError("百炼文本生成响应格式不正确") from error
        if not answer:
            raise RuntimeError("百炼文本生成返回了空回答")
        usage = result.get("usage") or {}
        return answer, {
            key: int(usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    @staticmethod
    def _evidence_block(number: int, result: SearchResult) -> str:
        source = result.chunk.metadata.get("source", result.chunk.document_id)
        section = result.chunk.metadata.get("section", f"Chunk {result.chunk.index}")
        return (
            f"[{number}] 来源：{source}；章节：{section}；"
            f"位置：{result.chunk.start}-{result.chunk.end}\n{result.chunk.content}"
        )

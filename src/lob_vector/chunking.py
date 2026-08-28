"""文档分块策略。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Chunk, Document


@dataclass(frozen=True, slots=True)
class FixedSizeChunker:
    """按字符数切分文档，并允许相邻 Chunk 保留重叠上下文。"""

    chunk_size: int = 500
    overlap: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.overlap < 0:
            raise ValueError("overlap 不能小于 0")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")

    def split(self, document: Document) -> list[Chunk]:
        """切分一份文档，返回包含原始字符位置的 Chunk。"""

        chunks: list[Chunk] = []
        start = 0
        step = self.chunk_size - self.overlap

        while start < len(document.content):
            end = min(start + self.chunk_size, len(document.content))
            content = document.content[start:end]

            if content.strip():
                index = len(chunks)
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{index}",
                        document_id=document.id,
                        content=content,
                        index=index,
                        start=start,
                        end=end,
                        metadata=document.metadata,
                    )
                )

            if end == len(document.content):
                break
            start += step

        return chunks

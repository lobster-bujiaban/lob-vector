"""统一向量存储抽象与内存实现。"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def clear(self) -> None: ...

    def count(self) -> int: ...

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
        self.upsert(chunks, vectors)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")
        prepared: list[tuple[Chunk, tuple[float, ...]]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            prepared.append((chunk, self._validate_vector(vector)))
        for chunk, vector in prepared:
            self._entries[chunk.id] = (chunk, vector)

    def delete(self, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            self._entries.pop(chunk_id, None)

    def clear(self) -> None:
        self._entries.clear()

    def count(self) -> int:
        return len(self._entries)

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


@dataclass(slots=True)
class ChromaVectorStore:
    """使用 Chroma PersistentClient 保存并检索调用方提供的向量。"""

    dimension: int
    path: Path | str = Path(".chroma")
    collection_name: str = "lob-vector"
    _client: object = field(init=False, repr=False)
    _collection: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension 必须是整数")
        if self.dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        if not self.collection_name.strip():
            raise ValueError("collection_name 不能为空")
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("使用 Chroma 前请先执行 uv sync") from error

        self.path = Path(self.path)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._open_collection()
        stored_dimension = (self._collection.metadata or {}).get("lob_dimension")
        if stored_dimension is not None and stored_dimension != self.dimension:
            raise ValueError(
                f"Chroma Collection 维度为 {stored_dimension}，当前 Embedder 为 {self.dimension}"
            )

    def _open_collection(self) -> object:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"lob_dimension": self.dimension, "description": "LOB Vector Stage 1"},
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self.upsert(chunks, vectors)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")
        if not chunks:
            return
        prepared_vectors = [self._validate_vector(vector) for vector in vectors]
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=[list(vector) for vector in prepared_vectors],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._collection.delete(ids=list(ids))

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()

    def count(self) -> int:
        return self._collection.count()

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(self._validate_vector(query_vector))],
            n_results=min(top_k, self.count()),
            where=self._where(metadata_filter),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        results = []
        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        ):
            metadata = dict(metadata or {})
            chunk = Chunk(
                id=chunk_id,
                document_id=str(metadata.pop("_lob_document_id")),
                content=content,
                index=int(metadata.pop("_lob_index")),
                start=int(metadata.pop("_lob_start")),
                end=int(metadata.pop("_lob_end")),
                metadata=metadata,
            )
            results.append(SearchResult(chunk=chunk, score=1.0 - float(distance), rank=rank))
        return results

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
    def _metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
        metadata = {key: value for key, value in chunk.metadata.items() if value is not None}
        metadata.update(
            {
                "_lob_document_id": chunk.document_id,
                "_lob_index": chunk.index,
                "_lob_start": chunk.start,
                "_lob_end": chunk.end,
            }
        )
        return metadata

    @staticmethod
    def _where(metadata_filter: MetadataFilter | None) -> dict[str, object] | None:
        if metadata_filter is None or not metadata_filter.conditions:
            return None
        operators = {"eq": "$eq", "ne": "$ne", "gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte"}
        clauses = [
            {condition.key: {operators[condition.operator]: condition.value}}
            for condition in metadata_filter.conditions
        ]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@dataclass(slots=True)
class QdrantVectorStore:
    """使用 Qdrant 本地持久化模式保存并检索调用方提供的向量。"""

    dimension: int
    path: Path | str = Path(".qdrant")
    collection_name: str = "lob-vector"
    url: str | None = None
    api_key: str | None = None
    _client: object = field(init=False, repr=False)

    @classmethod
    def from_environment(
        cls,
        dimension: int,
        *,
        path: Path | str = Path(".qdrant"),
        collection_name: str = "lob-vector",
    ) -> QdrantVectorStore:
        """根据 QDRANT_MODE 创建体验级本地存储或生产 Server 连接。"""
        mode = os.getenv("QDRANT_MODE", "local").strip().lower()
        if mode == "local":
            return cls(dimension, path, collection_name)
        if mode != "server":
            raise ValueError("QDRANT_MODE 只能是 local 或 server")

        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not url:
            raise RuntimeError("QDRANT_MODE=server 时必须设置 QDRANT_URL")
        if not api_key:
            raise RuntimeError("QDRANT_MODE=server 时必须设置 QDRANT_API_KEY")
        return cls(
            dimension,
            path,
            collection_name,
            url=url,
            api_key=api_key,
        )

    @property
    def mode(self) -> str:
        return "server" if self.url else "local"

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension 必须是整数")
        if self.dimension <= 0:
            raise ValueError("dimension 必须大于 0")
        if not self.collection_name.strip():
            raise ValueError("collection_name 不能为空")
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise RuntimeError("使用 Qdrant 前请先执行 uv sync") from error

        self.path = Path(self.path)
        if self.url:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self._client = QdrantClient(path=str(self.path))
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            collection = self._client.get_collection(self.collection_name)
            stored_dimension = collection.config.params.vectors.size
            if stored_dimension != self.dimension:
                raise ValueError(
                    f"Qdrant Collection 维度为 {stored_dimension}，当前 Embedder 为 {self.dimension}"
                )
        self._ensure_payload_indexes()

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        self.upsert(chunks, vectors)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量必须一致")
        if not chunks:
            return
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=self._point_id(chunk.id),
                vector=list(self._validate_vector(vector)),
                payload=self._payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        from qdrant_client import models

        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(item) for item in ids]),
            wait=True,
        )

    def clear(self) -> None:
        from qdrant_client import models

        self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
        )
        self._ensure_payload_indexes()

    def count(self) -> int:
        return int(self._client.count(collection_name=self.collection_name, exact=True).count)

    def close(self) -> None:
        self._client.close()

    def payload_indexes(self) -> dict[str, str]:
        """返回 Collection 中已实际建立的 Payload Index。"""
        schema = self._client.get_collection(self.collection_name).payload_schema
        indexes = {}
        for field_name, field_schema in schema.items():
            data_type = getattr(field_schema, "data_type", field_schema)
            indexes[field_name] = str(getattr(data_type, "value", data_type))
        return indexes

    def _ensure_payload_indexes(self) -> None:
        if not self.url:
            return
        from qdrant_client import models

        for field_name in ("tenant_id", "department", "permission"):
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 3,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if self.count() == 0:
            return []
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=list(self._validate_vector(query_vector)),
            query_filter=self._filter(metadata_filter),
            limit=top_k,
            with_payload=True,
        )
        results = []
        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})
            chunk = Chunk(
                id=str(payload.pop("_lob_chunk_id")),
                document_id=str(payload.pop("_lob_document_id")),
                content=str(payload.pop("_lob_content")),
                index=int(payload.pop("_lob_index")),
                start=int(payload.pop("_lob_start")),
                end=int(payload.pop("_lob_end")),
                metadata=payload,
            )
            results.append(SearchResult(chunk=chunk, score=float(point.score), rank=rank))
        return results

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
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lob-vector:{chunk_id}"))

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, str | int | float | bool | None]:
        payload = dict(chunk.metadata)
        payload.update(
            {
                "_lob_chunk_id": chunk.id,
                "_lob_document_id": chunk.document_id,
                "_lob_content": chunk.content,
                "_lob_index": chunk.index,
                "_lob_start": chunk.start,
                "_lob_end": chunk.end,
            }
        )
        return payload

    @staticmethod
    def _filter(metadata_filter: MetadataFilter | None) -> object | None:
        if metadata_filter is None or not metadata_filter.conditions:
            return None
        from qdrant_client import models

        conditions = []
        for condition in metadata_filter.conditions:
            if condition.operator == "eq":
                if condition.value is None:
                    conditions.append(
                        models.IsNullCondition(is_null=models.PayloadField(key=condition.key))
                    )
                else:
                    conditions.append(
                        models.FieldCondition(key=condition.key, match=models.MatchValue(value=condition.value))
                    )
            elif condition.operator == "ne":
                if condition.value is None:
                    excluded = models.IsNullCondition(
                        is_null=models.PayloadField(key=condition.key)
                    )
                else:
                    excluded = models.FieldCondition(
                        key=condition.key,
                        match=models.MatchValue(value=condition.value),
                    )
                conditions.append(
                    models.Filter(must_not=[excluded])
                )
            else:
                if isinstance(condition.value, bool) or not isinstance(
                    condition.value, (int, float)
                ):
                    raise ValueError("Qdrant 范围过滤值必须是数字")
                ranges = {
                    "gt": {"gt": condition.value},
                    "gte": {"gte": condition.value},
                    "lt": {"lt": condition.value},
                    "lte": {"lte": condition.value},
                }
                conditions.append(
                    models.FieldCondition(key=condition.key, range=models.Range(**ranges[condition.operator]))
                )
        return models.Filter(must=conditions)

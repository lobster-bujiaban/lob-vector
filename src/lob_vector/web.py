"""用于直观验证分块结果的本地 Web 实验台。"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from .chunking import FixedSizeChunker
from .embedding import BailianEmbedder, Embedder, HashEmbedder
from .generation import BailianChatGenerator, REFUSAL_TEXT
from .hnsw_experiment import run_hnsw_experiment
from .lightrag_client import LightRAGClient
from .milvus_experiment import run_milvus_index_experiment
from .models import Chunk, Document, SearchResult
from .retrieval import BM25Retriever, ranking_metrics, reciprocal_rank_fusion, rerank
from .vectorstore import ChromaVectorStore, MemoryVectorStore, MetadataCondition, MetadataFilter, MilvusVectorStore, QdrantVectorStore


_DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
_DATASET_DIRS = {
    "demo": _DATASETS_ROOT / "demo-knowledge-base",
    "shared": _DATASETS_ROOT / "knowledge-base",
}
_DATASET_METADATA = {
    "tech-stack.md": {"category": "profile", "topic": "engineering"},
    "ai-application-engineer-roadmap.md": {"category": "learning", "topic": "ai-application"},
}


def _embedder(payload: dict[str, Any], provider: str | None = None) -> Embedder:
    selected = provider or payload.get("embedder", "hash")
    if selected == "bailian":
        dimension = payload.get("dimension", 1024)
        if dimension == 32:  # Web 表单从 Hash 切换百炼时使用推荐维度。
            dimension = 1024
        return BailianEmbedder(
            dimension=dimension,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
    if selected != "hash":
        raise ValueError(f"不支持的 Embedder：{selected}")
    return HashEmbedder(payload.get("dimension", 32))


def _documents(payload: dict[str, Any]) -> list[Document]:
    source_mode = payload.get("source_mode", "text")
    if source_mode in {"dataset", "demo", "shared"}:
        dataset_name = "shared" if source_mode == "dataset" else source_mode
        dataset_dir = _DATASET_DIRS[dataset_name]
        documents = []
        for path in sorted(dataset_dir.glob("*.md")):
            if path.name == "README.md":
                continue
            content = path.read_text(encoding="utf-8")
            starts = [0, *(match.start() for match in re.finditer(r"(?m)^## ", content))]
            boundaries = sorted(set(starts)) + [len(content)]
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                section = content[start:end].strip()
                if not section:
                    continue
                heading = section.splitlines()[0].lstrip("# ")
                metadata = {
                    "source": f"datasets/{dataset_dir.name}/{path.name}",
                    "dataset": dataset_dir.name,
                    "section": heading,
                    "source_offset": start,
                    **_DATASET_METADATA.get(path.name, {}),
                }
                documents.append(Document(f"{path.stem}:{index}", section, metadata))
        if not documents:
            raise ValueError("知识库数据集为空")
        return documents

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入需要分块的文本")
    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("metadata 必须是 JSON 对象")
    return [Document("web-demo", text, raw_metadata)]


def _prepare(payload: dict[str, Any], provider: str | None = None) -> tuple[list[Chunk], list[list[float]], Embedder]:
    documents = _documents(payload)

    chunk_size = payload.get("chunk_size", 120)
    overlap = payload.get("overlap", 20)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("chunk_size 必须是整数")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise ValueError("overlap 必须是整数")
    chunker = FixedSizeChunker(chunk_size, overlap)
    chunks = [chunk for document in documents for chunk in chunker.split(document)]
    embedder = _embedder(payload, provider)
    vectors = embedder.embed([chunk.content for chunk in chunks])
    return chunks, vectors, embedder


def _chunk(payload: dict[str, Any]) -> dict[str, Any]:
    chunks, vectors, embedder = _prepare(payload)
    return {
        "document_length": sum(len(document.content) for document in _documents(payload)),
        "document_count": len(_documents(payload)),
        "chunk_count": len(chunks),
        "embedding_dimension": embedder.dimension,
        "embedder": payload.get("embedder", "hash"),
        "chunks": [
            {
                "id": chunk.id,
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "length": len(chunk.content),
                "content": chunk.content,
                "metadata": dict(chunk.metadata),
                "vector": vector,
                "vector_norm": sum(value * value for value in vector) ** 0.5,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    }


def _search(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("请输入查询问题")
    top_k = payload.get("top_k", 3)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    raw_filter = payload.get("filter", {})
    if not isinstance(raw_filter, dict):
        raise ValueError("filter 必须是 JSON 对象")
    conditions: list[MetadataCondition] = []
    for key, expected in raw_filter.items():
        if isinstance(expected, dict):
            if len(expected) != 1:
                raise ValueError(f"filter[{key!r}] 只能包含一个比较操作")
            operator, value = next(iter(expected.items()))
        else:
            operator, value = "eq", expected
        if operator not in {"eq", "ne", "gt", "gte", "lt", "lte"}:
            raise ValueError(f"不支持过滤操作：{operator}")
        conditions.append(MetadataCondition(str(key), operator, value))

    documents = _documents(payload)
    chunks, vectors, embedder = _prepare(payload)
    store = MemoryVectorStore(embedder.dimension)
    store.add(chunks, vectors)
    results = store.search(
        embedder.embed([query])[0],
        top_k=top_k,
        metadata_filter=MetadataFilter(tuple(conditions)) if conditions else None,
    )
    return {
        "query": query,
        "embedder": payload.get("embedder", "hash"),
        "embedding_dimension": embedder.dimension,
        "document_length": sum(len(document.content) for document in documents),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "result_count": len(results),
        "results": [
            {
                "rank": result.rank,
                "score": result.score,
                "id": result.chunk.id,
                "index": result.chunk.index,
                "start": result.chunk.start,
                "end": result.chunk.end,
                "content": result.chunk.content,
                "metadata": dict(result.chunk.metadata),
            }
            for result in results
        ],
    }


def _compare(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": payload.get("query"),
        "hash": _search({**payload, "embedder": "hash", "dimension": 32}),
        "bailian": _search({**payload, "embedder": "bailian", "dimension": 1024}),
    }


def _store_compare(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", "登录凭据想不起来，该怎样重新进入账号？")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("请输入查询问题")
    prepared_payload = {
        "source_mode": payload.get("source_mode", "demo"),
        "embedder": "hash",
        "dimension": 32,
        "chunk_size": payload.get("chunk_size", 300),
        "overlap": payload.get("overlap", 30),
    }
    chunks, vectors, embedder = _prepare(prepared_payload)
    query_vector = embedder.embed([query])[0]
    memory = MemoryVectorStore(embedder.dimension)
    memory.upsert(chunks, vectors)

    collection_name = "stage1-memory-chroma"
    persist_path = Path(".chroma") / "web-stage1"
    chroma = ChromaVectorStore(embedder.dimension, persist_path, collection_name)
    chroma.clear()
    lifecycle = [{"action": "clear", "count": chroma.count()}]
    chroma.upsert(chunks, vectors)
    lifecycle.append({"action": "upsert", "count": chroma.count()})
    chroma.upsert(chunks[:1], vectors[:1])
    lifecycle.append({"action": "update", "count": chroma.count()})
    chroma.delete([chunks[-1].id])
    lifecycle.append({"action": "delete", "count": chroma.count()})
    chroma.upsert(chunks[-1:], vectors[-1:])
    lifecycle.append({"action": "restore", "count": chroma.count()})
    reopened = ChromaVectorStore(embedder.dimension, persist_path, collection_name)
    lifecycle.append({"action": "reopen", "count": reopened.count()})

    top_k = payload.get("top_k", 3)
    memory_results = memory.search(query_vector, top_k=top_k)
    chroma_results = reopened.search(query_vector, top_k=top_k)
    memory_ids = [result.chunk.id for result in memory_results]
    chroma_ids = [result.chunk.id for result in chroma_results]
    score_deltas = [
        abs(left.score - right.score)
        for left, right in zip(memory_results, chroma_results, strict=True)
    ]

    def serialize(results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                "rank": result.rank,
                "score": result.score,
                "id": result.chunk.id,
                "content": result.chunk.content,
                "section": result.chunk.metadata.get("section"),
                "source": result.chunk.metadata.get("source"),
            }
            for result in results
        ]

    return {
        "query": query,
        "chunk_count": len(chunks),
        "dimension": embedder.dimension,
        "persist_path": str(persist_path),
        "collection": collection_name,
        "same_top_k": memory_ids == chroma_ids,
        "same_top_1": bool(memory_ids and memory_ids[0] == chroma_ids[0]),
        "max_score_delta": max(score_deltas, default=0.0),
        "memory": serialize(memory_results),
        "chroma": serialize(chroma_results),
        "lifecycle": lifecycle,
    }


def _qdrant_filter(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", "登录凭据想不起来，该怎样重新进入账号？")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("请输入查询问题")
    tenant_id = str(payload.get("tenant_id", "tenant-a"))
    department = str(payload.get("department", "support"))
    permission = str(payload.get("permission", "staff"))

    prepared = {
        "source_mode": "demo",
        "embedder": "hash",
        "dimension": 32,
        "chunk_size": 300,
        "overlap": 30,
    }
    chunks, vectors, embedder = _prepare(prepared)
    secured_chunks = [
        Chunk(
            id=f"tenant-a:{chunk.id}",
            document_id=chunk.document_id,
            content=chunk.content,
            index=chunk.index,
            start=chunk.start,
            end=chunk.end,
            metadata={
                **dict(chunk.metadata),
                "tenant_id": "tenant-a",
                "department": "support",
                "permission": "staff",
            },
        )
        for chunk in chunks
    ]
    forbidden = Document(
        id="tenant-b-secret",
        content=(
            f"{query}\n{query}\n{query}\n"
            "这是 tenant-b 的内部账号恢复手册，包含高度相关但禁止跨租户访问的机密流程。"
            "即使相似度最高，tenant-a 用户也绝不能召回本段。"
        ),
        metadata={
            "source": "tenant-b/private-account-recovery.md",
            "section": "内部账号恢复",
            "tenant_id": "tenant-b",
            "department": "security",
            "permission": "admin",
        },
    )
    forbidden_chunk = FixedSizeChunker(300, 30).split(forbidden)[0]
    all_chunks = [forbidden_chunk, *secured_chunks]
    all_vectors = [embedder.embed([forbidden_chunk.content])[0], *vectors]
    query_vector = embedder.embed([query])[0]
    metadata_filter = MetadataFilter(
        (
            MetadataCondition("tenant_id", "eq", tenant_id),
            MetadataCondition("department", "eq", department),
            MetadataCondition("permission", "eq", permission),
        )
    )

    store = QdrantVectorStore.from_environment(
        32,
        path=Path(".qdrant") / "web-stage2",
        collection_name="stage2-permission-lab",
    )
    try:
        storage_mode = store.mode
        store.clear()
        store.upsert(all_chunks, all_vectors)
        started = time.perf_counter()
        unfiltered = store.search(query_vector, top_k=3)
        unfiltered_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        filtered = store.search(query_vector, top_k=3, metadata_filter=metadata_filter)
        filtered_ms = (time.perf_counter() - started) * 1000
        indexed_count = store.count()
        payload_indexes = store.payload_indexes()
    finally:
        store.close()

    chroma = ChromaVectorStore(
        32,
        Path(".chroma") / "web-stage2",
        "stage2-product-compare",
    )
    chroma.clear()
    chroma.upsert(all_chunks, all_vectors)
    started = time.perf_counter()
    chroma_filtered = chroma.search(
        query_vector,
        top_k=3,
        metadata_filter=metadata_filter,
    )
    chroma_ms = (time.perf_counter() - started) * 1000

    def serialize(results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                "rank": result.rank,
                "score": result.score,
                "id": result.chunk.id,
                "content": result.chunk.content,
                "source": result.chunk.metadata.get("source"),
                "tenant_id": result.chunk.metadata.get("tenant_id"),
                "department": result.chunk.metadata.get("department"),
                "permission": result.chunk.metadata.get("permission"),
            }
            for result in results
        ]

    unfiltered_ids = {result.chunk.id for result in unfiltered}
    filtered_ids = {result.chunk.id for result in filtered}
    blocked = [result for result in unfiltered if result.chunk.id not in filtered_ids]
    tenant_counts: dict[str, int] = {}
    for chunk in all_chunks:
        tenant = str(chunk.metadata.get("tenant_id"))
        tenant_counts[tenant] = tenant_counts.get(tenant, 0) + 1
    return {
        "query": query,
        "storage_mode": storage_mode,
        "filter": {
            "tenant_id": tenant_id,
            "department": department,
            "permission": permission,
        },
        "indexed_count": indexed_count,
        "payload_indexes": payload_indexes,
        "expected_payload_indexes": [
            "tenant_id",
            "department",
            "permission",
            "created_at",
        ],
        "tenant_counts": tenant_counts,
        "blocked_count": len(unfiltered_ids - filtered_ids),
        "safe": all(
            result.chunk.metadata.get("tenant_id") == tenant_id
            and result.chunk.metadata.get("department") == department
            and result.chunk.metadata.get("permission") == permission
            for result in filtered
        ),
        "unfiltered_ms": unfiltered_ms,
        "filtered_ms": filtered_ms,
        "chroma_ms": chroma_ms,
        "unfiltered": serialize(unfiltered),
        "blocked": serialize(blocked),
        "filtered": serialize(filtered),
        "chroma_filtered": serialize(chroma_filtered),
        "same_filtered_top_k": [item.chunk.id for item in filtered]
        == [item.chunk.id for item in chroma_filtered],
    }


def _rag_answer(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("请输入需要回答的问题")
    top_k = payload.get("top_k", 4)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 8:
        raise ValueError("top_k 必须是 1 到 8 的整数")
    min_score = payload.get("min_score", 0.2)
    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
        raise ValueError("min_score 必须是数字")
    store_name = payload.get("store", "qdrant")
    if store_name not in {"memory", "chroma", "qdrant", "milvus"}:
        raise ValueError("RAG store 只能是 memory、chroma、qdrant 或 milvus")

    prepared = {
        "source_mode": "demo",
        "embedder": "bailian",
        "dimension": 1024,
        "chunk_size": 300,
        "overlap": 30,
    }


_RETRIEVAL_EVALUATION = (
    ("登录凭据想不起来，该怎样重新进入账号？", "01-account-security.md", "忘记密码"),
    ("接口提示访问过于频繁，客户端接下来该怎么做？", "05-api-rate-limit.md", "请求限流"),
    ("为什么内容再相似也不能返回其他客户的资料？", "07-tenant-permission.md", "检索隔离"),
    ("热门数据刚失效，大量流量压到数据库怎么处理？", "06-cache-failures.md", "热点键失效"),
    ("设备总是断开连接，需要检查哪些因素？", "04-device-offline.md", "设备频繁掉线"),
    ("模型或提示词改完，怎么证明效果真的变好了？", "10-observability.md", "质量评估"),
)


def _retrieval_compare(payload: dict[str, Any]) -> dict[str, Any]:
    query = payload.get("query", _RETRIEVAL_EVALUATION[0][0])
    if not isinstance(query, str) or not query.strip():
        raise ValueError("请输入检索问题")
    provider = payload.get("embedder", "hash")
    if provider not in {"hash", "bailian"}:
        raise ValueError("embedder 只能是 hash 或 bailian")
    prepared = {
        "source_mode": "demo",
        "embedder": provider,
        "dimension": 1024 if provider == "bailian" else 32,
        "chunk_size": 300,
        "overlap": 30,
    }
    chunks, vectors, embedder = _prepare(prepared)
    questions = [item[0] for item in _RETRIEVAL_EVALUATION]
    all_questions = [query, *questions]
    query_vectors = embedder.embed(all_questions)
    vector_store = MemoryVectorStore(embedder.dimension)
    vector_store.upsert(chunks, vectors)
    bm25 = BM25Retriever(chunks)

    rankings: dict[str, list[list[SearchResult]]] = {
        "vector": [],
        "bm25": [],
        "hybrid": [],
        "reranked": [],
    }
    latencies: dict[str, list[float]] = {name: [] for name in rankings}
    current: dict[str, list[SearchResult]] = {}
    for index, (question, vector) in enumerate(
        zip(all_questions, query_vectors, strict=True)
    ):
        started = time.perf_counter()
        vector_results = vector_store.search(vector, top_k=10)
        latencies["vector"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        bm25_results = bm25.search(question, top_k=10)
        latencies["bm25"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        hybrid_results = reciprocal_rank_fusion(
            (vector_results, bm25_results), top_k=10
        )
        latencies["hybrid"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        reranked_results = rerank(question, hybrid_results, top_k=10)
        latencies["reranked"].append((time.perf_counter() - started) * 1000)
        result_set = {
            "vector": vector_results,
            "bm25": bm25_results,
            "hybrid": hybrid_results,
            "reranked": reranked_results,
        }
        if index == 0:
            current = result_set
        else:
            for name, results in result_set.items():
                rankings[name].append(results)

    expected = [(item[1], item[2]) for item in _RETRIEVAL_EVALUATION]

    def serialize(results: list[SearchResult]) -> list[dict[str, Any]]:
        return [
            {
                "rank": result.rank,
                "score": result.score,
                "content": result.chunk.content,
                "source": result.chunk.metadata.get("source"),
                "section": result.chunk.metadata.get("section"),
            }
            for result in results[:3]
        ]

    return {
        "query": query,
        "embedder": provider,
        "evaluation_count": len(_RETRIEVAL_EVALUATION),
        "results": {name: serialize(results) for name, results in current.items()},
        "metrics": {
            name: {
                **ranking_metrics(results, expected, top_k=3),
                "average_ms": statistics.mean(latencies[name]),
            }
            for name, results in rankings.items()
        },
    }
    retrieval_started = time.perf_counter()
    chunks, vectors, embedder = _prepare(prepared)
    if store_name == "chroma":
        store = ChromaVectorStore(
            embedder.dimension,
            Path(".chroma") / "web-stage4",
            "stage4-rag",
        )
    elif store_name == "qdrant":
        store = QdrantVectorStore.from_environment(
            embedder.dimension,
            path=Path(".qdrant") / "web-stage4",
            collection_name="stage4-rag",
        )
    elif store_name == "milvus":
        store = MilvusVectorStore.from_environment(
            embedder.dimension,
            path=Path(".milvus/stage4-rag.db"),
            collection_name="stage4_rag",
        )
    else:
        store = MemoryVectorStore(embedder.dimension)
    try:
        store.clear()
        store.upsert(chunks, vectors)
        candidates = store.search(embedder.embed([question])[0], top_k=top_k)
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()
    evidence = [result for result in candidates if result.score >= float(min_score)]
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    generation_started = time.perf_counter()
    if evidence:
        generator = BailianChatGenerator(
            model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus"),
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        answer, usage = generator.generate(question, evidence)
    else:
        answer, usage = REFUSAL_TEXT, {}
    generation_ms = (time.perf_counter() - generation_started) * 1000
    cited_numbers = sorted(
        {
            int(number)
            for number in re.findall(r"\[(\d+)\]", answer)
            if 1 <= int(number) <= len(evidence)
        }
    )
    if answer != REFUSAL_TEXT and not cited_numbers:
        answer = REFUSAL_TEXT
    return {
        "question": question,
        "store": store_name,
        "answer": answer,
        "refused": answer == REFUSAL_TEXT,
        "top_k": top_k,
        "min_score": min_score,
        "candidate_count": len(candidates),
        "evidence_count": len(evidence),
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "usage": usage,
        "cited_numbers": cited_numbers,
        "evidence": [
            {
                "number": number,
                "score": result.score,
                "content": result.chunk.content,
                "source": result.chunk.metadata.get("source", result.chunk.document_id),
                "section": result.chunk.metadata.get(
                    "section", f"Chunk {result.chunk.index}"
                ),
                "start": result.chunk.start,
                "end": result.chunk.end,
                "cited": number in cited_numbers,
            }
            for number, result in enumerate(evidence, start=1)
        ],
    }


def _dataset(source_mode: str = "demo") -> dict[str, Any]:
    documents = _documents({"source_mode": source_mode})
    sources: dict[str, dict[str, Any]] = {}
    for document in documents:
        source = str(document.metadata["source"])
        entry = sources.setdefault(
            source,
            {"source": source, "section_count": 0, "metadata": dict(document.metadata)},
        )
        entry["section_count"] += 1
    for entry in sources.values():
        entry["metadata"].pop("section", None)
        entry["metadata"].pop("source_offset", None)
    return {
        "name": "典型知识库对照集" if source_mode == "demo" else "共享知识库实验集",
        "document_count": len(documents),
        "source_count": len(sources),
        "character_count": sum(len(document.content) for document in documents),
        "sources": list(sources.values()),
    }


def _lightrag_index(_: dict[str, Any]) -> dict[str, Any]:
    documents = _documents({"source_mode": "demo"})
    texts = [
        f"来源：{document.metadata['source']}\n章节：{document.metadata['section']}\n\n{document.content}"
        for document in documents
    ]
    client = LightRAGClient.from_env()
    health = client.health()
    accepted = client.insert_texts(texts)
    return {
        "document_count": len(texts),
        "character_count": sum(map(len, texts)),
        "health": {
            key: health.get(key)
            for key in ("status", "core_version", "api_version", "pipeline_busy")
        },
        "accepted": accepted,
        "note": "LightRAG 会异步抽取实体和关系；控制台显示全部文档处理完成后再运行查询对照。",
    }


def _lightrag_compare(payload: dict[str, Any]) -> dict[str, Any]:
    question = payload.get("question", "多个系统出现故障时，限流、缓存和可观测性之间有什么关系？")
    if not isinstance(question, str) or len(question.strip()) < 3:
        raise ValueError("请输入至少 3 个字符的问题")
    client = LightRAGClient.from_env()
    rows = []
    for mode in ("naive", "local", "global", "mix"):
        result = client.query(question.strip(), mode)
        references = result.get("references") or []
        rows.append(
            {
                "mode": mode,
                "answer": str(result.get("response", "")),
                "elapsed_ms": result["elapsed_ms"],
                "references": references if isinstance(references, list) else [],
            }
        )
    return {"question": question.strip(), "rows": rows}


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/api/dataset", "/api/dataset/demo"}:
            self._send_json(HTTPStatus.OK, _dataset("demo"))
            return
        if self.path == "/api/dataset/shared":
            self._send_json(HTTPStatus.OK, _dataset("shared"))
            return
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = files("lob_vector.static").joinpath("index.html").read_bytes()
        self._send(HTTPStatus.OK, content, "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/chunk", "/api/search", "/api/compare", "/api/store-compare", "/api/qdrant-filter", "/api/hnsw-experiment", "/api/milvus-index-experiment", "/api/rag-answer", "/api/retrieval-compare", "/api/lightrag-index", "/api/lightrag-compare"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("演示文本不能超过 1 MB")
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/search":
                result = _search(payload)
            elif self.path == "/api/compare":
                result = _compare(payload)
            elif self.path == "/api/store-compare":
                result = _store_compare(payload)
            elif self.path == "/api/qdrant-filter":
                result = _qdrant_filter(payload)
            elif self.path == "/api/hnsw-experiment":
                result = run_hnsw_experiment(
                    point_count=payload.get("point_count", 10_000),
                    query_count=payload.get("query_count", 12),
                )
            elif self.path == "/api/milvus-index-experiment":
                result = run_milvus_index_experiment(
                    point_count=payload.get("point_count", 10_000),
                    query_count=payload.get("query_count", 12),
                )
            elif self.path == "/api/rag-answer":
                result = _rag_answer(payload)
            elif self.path == "/api/retrieval-compare":
                result = _retrieval_compare(payload)
            elif self.path == "/api/lightrag-index":
                result = _lightrag_index(payload)
            elif self.path == "/api/lightrag-compare":
                result = _lightrag_compare(payload)
            else:
                result = _chunk(payload)
            self._send_json(HTTPStatus.OK, result)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except RuntimeError as error:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, content, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"LOB Vector 检索实验台：http://{host}:{port}")
    if os.getenv("QDRANT_MODE", "local").strip().lower() == "server":
        qdrant_url = os.getenv("QDRANT_URL", "").strip().rstrip("/")
        if qdrant_url:
            print(f"Qdrant 控制台：{qdrant_url}/dashboard")
    if os.getenv("MILVUS_MODE", "lite").strip().lower() == "server":
        print(
            "Milvus 控制台："
            f"http://127.0.0.1:{os.getenv('MILVUS_WEBUI_PORT', '9091')}/webui/"
        )
        print(
            "MinIO 控制台："
            f"http://127.0.0.1:{os.getenv('MILVUS_MINIO_CONSOLE_PORT', '19001')}"
        )
    if os.getenv("LIGHTRAG_MODE", "disabled").strip().lower() == "server":
        light_url = os.getenv("LIGHTRAG_URL", "http://127.0.0.1:9621").rstrip("/")
        print(f"LightRAG 控制台：{light_url}/webui")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

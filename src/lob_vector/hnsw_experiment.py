"""Qdrant Server HNSW 参数与 Segment 生命周期实验。"""

from __future__ import annotations

import math
import os
import random
import statistics
import time
from typing import Any, Sequence


def run_hnsw_experiment(
    *,
    point_count: int = 10_000,
    query_count: int = 12,
    dimension: int = 64,
) -> dict[str, Any]:
    """构造可复现数据，用精确搜索评估不同 hnsw_ef。"""
    if os.getenv("QDRANT_MODE", "local").strip().lower() != "server":
        raise RuntimeError(
            "HNSW 实验必须使用 Qdrant Server：请设置 QDRANT_MODE=server、QDRANT_URL 和 QDRANT_API_KEY"
        )
    if point_count not in {5_000, 10_000, 20_000}:
        raise ValueError("point_count 只支持 5000、10000 或 20000")
    if query_count not in {8, 12, 20}:
        raise ValueError("query_count 只支持 8、12 或 20")

    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url or not api_key:
        raise RuntimeError("HNSW 实验需要 QDRANT_URL 和 QDRANT_API_KEY")

    from qdrant_client import QdrantClient, models

    collection = "stage2-hnsw-lab"
    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=60,
        trust_env=False,
    )
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100,
                default_segment_number=2,
            ),
        )

        seed = 20260830
        rng = random.Random(seed)
        vectors = [_unit_vector(rng, dimension) for _ in range(point_count)]
        query_indexes = rng.sample(range(point_count), query_count)
        queries = [
            _normalize(
                [value + rng.gauss(0.0, 0.08) for value in vectors[index]]
            )
            for index in query_indexes
        ]

        build_started = time.perf_counter()
        for start in range(0, point_count, 500):
            batch = [
                models.PointStruct(id=index, vector=vectors[index])
                for index in range(start, min(start + 500, point_count))
            ]
            client.upsert(collection_name=collection, points=batch, wait=True)
        write_ms = (time.perf_counter() - build_started) * 1000

        index_wait_started = time.perf_counter()
        info = client.get_collection(collection)
        deadline = time.monotonic() + 45
        while (info.indexed_vectors_count or 0) < point_count * 0.9:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
            info = client.get_collection(collection)
        index_wait_ms = (time.perf_counter() - index_wait_started) * 1000

        exact_ids, exact_latencies = _query_batch(
            client,
            models,
            collection,
            queries,
            top_k=10,
            exact=True,
        )
        rows = [
            {
                "mode": "exact",
                "hnsw_ef": None,
                "recall_at_10": 1.0,
                **_latency_metrics(exact_latencies),
            }
        ]
        for hnsw_ef in (16, 64, 128):
            # 一次预热，避免首次连接与缓存建立影响正式计时。
            client.query_points(
                collection_name=collection,
                query=queries[0],
                search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=False),
                limit=10,
            )
            approximate_ids, latencies = _query_batch(
                client,
                models,
                collection,
                queries,
                top_k=10,
                hnsw_ef=hnsw_ef,
            )
            recall = statistics.mean(
                len(set(expected) & set(actual)) / 10
                for expected, actual in zip(exact_ids, approximate_ids, strict=True)
            )
            rows.append(
                {
                    "mode": "hnsw",
                    "hnsw_ef": hnsw_ef,
                    "recall_at_10": recall,
                    **_latency_metrics(latencies),
                }
            )

        before = client.get_collection(collection)
        changed_ids = list(range(min(100, point_count)))
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=changed_ids),
            wait=True,
        )
        after_delete = client.get_collection(collection)
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=index, vector=vectors[index])
                for index in changed_ids
            ],
            wait=True,
        )
        after_restore = client.get_collection(collection)

        current = client.get_collection(collection)
        return {
            "collection": collection,
            "seed": seed,
            "point_count": point_count,
            "query_count": query_count,
            "dimension": dimension,
            "top_k": 10,
            "write_ms": write_ms,
            "index_wait_ms": index_wait_ms,
            "indexed_vectors_count": current.indexed_vectors_count or 0,
            "segments_count": current.segments_count,
            "status": str(getattr(current.status, "value", current.status)),
            "hnsw": {"m": 16, "ef_construct": 100, "full_scan_threshold": 10},
            "rows": rows,
            "lifecycle": [
                _collection_step("写入完成", before),
                _collection_step("删除 100 条", after_delete),
                _collection_step("恢复 100 条", after_restore),
            ],
            "warning": (
                None
                if (current.indexed_vectors_count or 0) >= point_count * 0.9
                else "后台 HNSW 索引未在 45 秒内完成，本次近似查询可能仍包含未索引 Segment"
            ),
        }
    finally:
        client.close()


def _unit_vector(rng: random.Random, dimension: int) -> list[float]:
    return _normalize([rng.gauss(0.0, 1.0) for _ in range(dimension)])


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _query_batch(
    client: Any,
    models: Any,
    collection: str,
    queries: Sequence[Sequence[float]],
    *,
    top_k: int,
    exact: bool = False,
    hnsw_ef: int | None = None,
) -> tuple[list[list[int | str]], list[float]]:
    ids = []
    latencies = []
    for query in queries:
        started = time.perf_counter()
        response = client.query_points(
            collection_name=collection,
            query=list(query),
            search_params=models.SearchParams(
                exact=exact,
                hnsw_ef=hnsw_ef,
            ),
            limit=top_k,
            with_payload=False,
            with_vectors=False,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        ids.append([point.id for point in response.points])
    return ids, latencies


def _latency_metrics(latencies: Sequence[float]) -> dict[str, float]:
    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "average_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
    }


def _collection_step(action: str, info: Any) -> dict[str, Any]:
    return {
        "action": action,
        "points": info.points_count or 0,
        "indexed": info.indexed_vectors_count or 0,
        "segments": info.segments_count,
        "status": str(getattr(info.status, "value", info.status)),
    }

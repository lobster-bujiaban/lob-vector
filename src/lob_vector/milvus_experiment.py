"""Milvus Standalone 索引类型对照实验。"""

from __future__ import annotations

import math
import os
import random
import statistics
import time
from typing import Any, Sequence


def run_milvus_index_experiment(
    *, point_count: int = 10_000, query_count: int = 12, dimension: int = 64
) -> dict[str, Any]:
    if os.getenv("MILVUS_MODE", "lite").strip().lower() != "server":
        raise RuntimeError(
            "索引对照必须使用 Milvus Standalone：请设置 MILVUS_MODE=server 和 MILVUS_URI"
        )
    if point_count not in {5_000, 10_000, 20_000}:
        raise ValueError("point_count 只支持 5000、10000 或 20000")
    if query_count not in {8, 12, 20}:
        raise ValueError("query_count 只支持 8、12 或 20")

    uri = os.getenv("MILVUS_URI", "").strip()
    if not uri:
        raise RuntimeError("MILVUS_MODE=server 时必须设置 MILVUS_URI")
    if not uri.startswith(("http://", "https://")):
        raise ValueError("MILVUS_MODE=server 时 MILVUS_URI 必须是 HTTP(S) 地址")

    from pymilvus import DataType, MilvusClient

    client = MilvusClient(
        uri=uri,
        token=os.getenv("MILVUS_TOKEN", "").strip() or None,
        timeout=60,
    )
    seed = 20260830
    rng = random.Random(seed)
    vectors = [_unit_vector(rng, dimension) for _ in range(point_count)]
    query_indexes = rng.sample(range(point_count), query_count)
    queries = [
        _normalize([value + rng.gauss(0.0, 0.08) for value in vectors[index]])
        for index in query_indexes
    ]
    rows = []
    ground_truth: list[list[int]] | None = None
    definitions = [
        ("FLAT", {}, {}),
        ("IVF_FLAT", {"nlist": 128}, {"nprobe": 16}),
        ("HNSW", {"M": 16, "efConstruction": 100}, {"ef": 64}),
    ]
    try:
        for index_type, build_params, search_params in definitions:
            collection = f"stage3_{index_type.lower()}"
            if client.has_collection(collection):
                client.drop_collection(collection)
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(
                field_name="vector",
                index_type=index_type,
                metric_type="COSINE",
                params=build_params,
            )
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=dimension,
            )
            build_started = time.perf_counter()
            client.create_collection(
                collection_name=collection,
                schema=schema,
                index_params=indexes,
            )
            for start in range(0, point_count, 500):
                client.insert(
                    collection_name=collection,
                    data=[
                        {"id": index, "vector": vectors[index]}
                        for index in range(start, min(start + 500, point_count))
                    ],
                )
            client.flush(collection)
            client.load_collection(collection)
            build_ms = (time.perf_counter() - build_started) * 1000
            index_name = client.list_indexes(collection)[0]
            actual_index = client.describe_index(collection, index_name)
            actual_index_type = str(actual_index.get("index_type"))
            if actual_index_type != index_type:
                raise RuntimeError(
                    f"Milvus 实际创建了 {actual_index_type}，预期为 {index_type}"
                )
            load_state = client.get_load_state(collection)["state"]

            ids, latencies = _search_batch(
                client,
                collection,
                queries,
                search_params=search_params,
            )
            if ground_truth is None:
                ground_truth = ids
                recall = 1.0
            else:
                recall = statistics.mean(
                    len(set(expected) & set(actual)) / 10
                    for expected, actual in zip(ground_truth, ids, strict=True)
                )
            rows.append(
                {
                    "index_type": index_type,
                    "collection": collection,
                    "actual_index_type": actual_index_type,
                    "index_state": str(actual_index.get("state")),
                    "indexed_rows": int(actual_index.get("indexed_rows", 0)),
                    "load_state": str(getattr(load_state, "name", load_state)),
                    "build_params": build_params,
                    "search_params": search_params,
                    "build_ms": build_ms,
                    "recall_at_10": recall,
                    **_latency_metrics(latencies),
                }
            )

        return {
            "seed": seed,
            "point_count": point_count,
            "query_count": query_count,
            "dimension": dimension,
            "top_k": 10,
            "rows": rows,
            "architecture": [
                {"name": "Proxy", "role": "接收请求并合并各节点结果"},
                {"name": "Coordinator", "role": "调度 Collection、Segment、索引和负载"},
                {"name": "Query Node", "role": "加载索引并执行向量查询"},
                {"name": "Data Node", "role": "执行 Compaction 与离线索引构建"},
                {"name": "etcd", "role": "保存 Collection Schema 与系统元数据"},
                {"name": "MinIO", "role": "保存日志、原始数据和索引文件"},
            ],
        }
    finally:
        client.close()


def _unit_vector(rng: random.Random, dimension: int) -> list[float]:
    return _normalize([rng.gauss(0.0, 1.0) for _ in range(dimension)])


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _search_batch(
    client: Any,
    collection: str,
    queries: Sequence[Sequence[float]],
    *,
    search_params: dict[str, int],
) -> tuple[list[list[int]], list[float]]:
    ids = []
    latencies = []
    for query in queries:
        started = time.perf_counter()
        response = client.search(
            collection_name=collection,
            data=[list(query)],
            limit=10,
            search_params={"metric_type": "COSINE", "params": search_params},
            output_fields=[],
        )[0]
        latencies.append((time.perf_counter() - started) * 1000)
        ids.append([int(hit["id"]) for hit in response])
    return ids, latencies


def _latency_metrics(latencies: Sequence[float]) -> dict[str, float]:
    ordered = sorted(latencies)
    return {
        "average_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
    }

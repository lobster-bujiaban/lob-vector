"""用于直观验证分块结果的本地 Web 实验台。"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

from .chunking import FixedSizeChunker
from .embedding import HashEmbedder
from .models import Document
from .vectorstore import MemoryVectorStore, MetadataCondition, MetadataFilter


def _chunk(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入需要分块的文本")

    chunk_size = payload.get("chunk_size", 120)
    overlap = payload.get("overlap", 20)
    dimension = payload.get("dimension", 32)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("chunk_size 必须是整数")
    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise ValueError("overlap 必须是整数")
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise ValueError("dimension 必须是整数")

    raw_metadata = payload.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("metadata 必须是 JSON 对象")

    document = Document("web-demo", text, raw_metadata)
    chunks = FixedSizeChunker(chunk_size, overlap).split(document)
    embedder = HashEmbedder(dimension)
    vectors = embedder.embed([chunk.content for chunk in chunks])
    return {
        "document_length": len(text),
        "chunk_count": len(chunks),
        "embedding_dimension": embedder.dimension,
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
    chunk_result = _chunk(payload)
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

    chunks = []
    vectors = []
    for item in chunk_result["chunks"]:
        document = Document("web-demo", payload["text"], payload.get("metadata", {}))
        chunk = FixedSizeChunker(payload.get("chunk_size", 120), payload.get("overlap", 20)).split(document)[item["index"]]
        chunks.append(chunk)
        vectors.append(item["vector"])

    dimension = chunk_result["embedding_dimension"]
    embedder = HashEmbedder(dimension)
    store = MemoryVectorStore(dimension)
    store.add(chunks, vectors)
    results = store.search(
        embedder.embed([query])[0],
        top_k=top_k,
        metadata_filter=MetadataFilter(tuple(conditions)) if conditions else None,
    )
    return {
        "query": query,
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


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = files("lob_vector.static").joinpath("index.html").read_bytes()
        self._send(HTTPStatus.OK, content, "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/chunk", "/api/search"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("演示文本不能超过 1 MB")
            payload = json.loads(self.rfile.read(length))
            result = _search(payload) if self.path == "/api/search" else _chunk(payload)
            self._send_json(HTTPStatus.OK, result)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

"""LOB Vector 命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from . import __version__
from .chunking import FixedSizeChunker
from .embedding import BailianEmbedder, Embedder, HashEmbedder
from .models import Document
from .vectorstore import MemoryVectorStore, MetadataCondition, MetadataFilter
from .web import serve


def _load_dotenv(path: Path = Path(".env")) -> None:
    """读取简单的 KEY=VALUE 配置，已存在的环境变量优先。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise argparse.ArgumentTypeError(
                f"metadata 必须使用 key=value 格式，当前值：{item!r}"
            )
        metadata[key.strip()] = value
    return metadata


def _chunk_command(args: argparse.Namespace) -> None:
    path: Path = args.file
    content = path.read_text(encoding="utf-8")
    metadata = _metadata(args.metadata)
    metadata.setdefault("source", str(path))

    document = Document(
        id=args.document_id or path.stem,
        content=content,
        metadata=metadata,
    )
    chunks = FixedSizeChunker(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    ).split(document)

    output = [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "index": chunk.index,
            "start": chunk.start,
            "end": chunk.end,
            "content": chunk.content,
            "metadata": dict(chunk.metadata),
        }
        for chunk in chunks
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _web_command(args: argparse.Namespace) -> None:
    serve(host=args.host, port=args.port)


def _embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "bailian":
        return BailianEmbedder(
            dimension=args.dimension or 1024,
            model=args.model,
            base_url=args.base_url,
        )
    return HashEmbedder(dimension=args.dimension or 32)


def _embed_command(args: argparse.Namespace) -> None:
    embedder = _embedder(args)
    vectors = embedder.embed(args.text)
    output = [
        {
            "text": text,
            "dimension": embedder.dimension,
            "norm": sum(value * value for value in vector) ** 0.5,
            "vector": vector,
        }
        for text, vector in zip(args.text, vectors, strict=True)
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _scalar(value: str) -> str | int | float | bool | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, (str, int, float, bool, type(None))) else value


def _where(values: Sequence[str]) -> MetadataFilter | None:
    operators = ((">=", "gte"), ("<=", "lte"), ("!=", "ne"), ("=", "eq"), (">", "gt"), ("<", "lt"))
    conditions: list[MetadataCondition] = []
    for item in values:
        for symbol, operator in operators:
            key, separator, value = item.partition(symbol)
            if separator and key.strip() and value.strip():
                conditions.append(MetadataCondition(key.strip(), operator, _scalar(value.strip())))
                break
        else:
            raise argparse.ArgumentTypeError(
                f"where 格式无效：{item!r}，示例：year>=2025"
            )
    return MetadataFilter(tuple(conditions)) if conditions else None


def _search_command(args: argparse.Namespace) -> None:
    embedder = _embedder(args)
    chunker = FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    chunks = []
    for path in args.files:
        document = Document(
            id=str(path),
            content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path)},
        )
        chunks.extend(chunker.split(document))

    store = MemoryVectorStore(dimension=embedder.dimension)
    store.add(chunks, embedder.embed([chunk.content for chunk in chunks]))
    results = store.search(
        embedder.embed([args.query])[0],
        top_k=args.top_k,
        metadata_filter=_where(args.where),
    )
    output = [
        {
            "rank": result.rank,
            "score": result.score,
            "content": result.chunk.content,
            "source": result.chunk.metadata.get("source"),
            "document_id": result.chunk.document_id,
            "chunk_id": result.chunk.id,
            "index": result.chunk.index,
            "start": result.chunk.start,
            "end": result.chunk.end,
            "metadata": dict(result.chunk.metadata),
        }
        for result in results
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lob-vector",
        description="从零学习向量检索与 RAG",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    chunk_parser = subparsers.add_parser("chunk", help="按固定字符数切分文本文件")
    chunk_parser.add_argument("file", type=Path, help="UTF-8 文本文件路径")
    chunk_parser.add_argument("--document-id", help="文档 ID，默认使用文件名")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="每个 Chunk 的最大字符数，默认 500",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="相邻 Chunk 重叠字符数，默认 50",
    )
    chunk_parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="附加 Metadata，可重复传入",
    )
    chunk_parser.set_defaults(handler=_chunk_command)

    embed_parser = subparsers.add_parser("embed", help="生成确定性的本地 Hash Embedding")
    embed_parser.add_argument("text", nargs="+", help="需要转换为向量的文本")
    _add_embedder_arguments(embed_parser)
    embed_parser.set_defaults(handler=_embed_command)

    search_parser = subparsers.add_parser("search", help="在本地文本文件中执行向量检索")
    search_parser.add_argument("query", help="查询文本")
    search_parser.add_argument("files", nargs="+", type=Path, help="UTF-8 文本文件")
    search_parser.add_argument("--top-k", type=int, default=3, help="返回结果数，默认 3")
    _add_embedder_arguments(search_parser)
    search_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk 最大字符数")
    search_parser.add_argument("--overlap", type=int, default=50, help="相邻 Chunk 重叠字符数")
    search_parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="CONDITION",
        help="Metadata 条件，可重复传入，例如 source=README.md 或 year>=2025",
    )
    search_parser.set_defaults(handler=_search_command)

    web_parser = subparsers.add_parser("web", help="启动可视化分块实验台")
    web_parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    web_parser.add_argument("--port", type=int, default=8765, help="监听端口")
    web_parser.set_defaults(handler=_web_command)
    return parser


def _add_embedder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedder",
        choices=("hash", "bailian"),
        default="hash",
        help="Embedding 实现，默认 hash",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="向量维度；hash 默认 32，bailian 默认 1024",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-v4",
        help="百炼模型名称，默认 text-embedding-v4",
    )
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="百炼 OpenAI 兼容接口地址",
    )


def main() -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()

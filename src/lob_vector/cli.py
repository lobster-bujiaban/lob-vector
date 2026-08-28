"""LOB Vector 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .chunking import FixedSizeChunker
from .embedding import HashEmbedder
from .models import Document
from .web import serve


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


def _embed_command(args: argparse.Namespace) -> None:
    embedder = HashEmbedder(dimension=args.dimension)
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
    embed_parser.add_argument(
        "--dimension",
        type=int,
        default=32,
        help="向量维度，默认 32",
    )
    embed_parser.set_defaults(handler=_embed_command)

    web_parser = subparsers.add_parser("web", help="启动可视化分块实验台")
    web_parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    web_parser.add_argument("--port", type=int, default=8765, help="监听端口")
    web_parser.set_defaults(handler=_web_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()

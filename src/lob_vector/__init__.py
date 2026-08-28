"""LOB Vector 核心包。"""

from .chunking import FixedSizeChunker
from .embedding import Embedder, HashEmbedder
from .models import Chunk, Document, Metadata, MetadataValue, SearchResult
from .vectorstore import MemoryVectorStore, MetadataCondition, MetadataFilter, VectorStore

__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "FixedSizeChunker",
    "HashEmbedder",
    "Metadata",
    "MetadataValue",
    "MemoryVectorStore",
    "MetadataCondition",
    "MetadataFilter",
    "SearchResult",
    "VectorStore",
]

__version__ = "0.1.0"

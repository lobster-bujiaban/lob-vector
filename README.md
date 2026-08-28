# LOB Vector

从零掌握向量检索与 RAG 的核心链路，并通过 Chroma、Qdrant、Milvus、LightRAG 和 RAGFlow 验证工程实现。

本项目采用与 LOB Codex 相似的学习方法：围绕一条真实主链路分阶段实现，每个阶段都保留可运行、可观察、可对照的结果。区别在于，本项目不以完整复刻某一个产品为目标，而是先手写核心算法，再接入不同向量数据库，最后形成完整 RAG 系统。

## 核心目标

- 理解文档、Chunk、Embedding、距离度量和 TopK 召回之间的关系。
- 掌握 Metadata Filter、HNSW、Payload Index、分片、索引和持久化。
- 实现向量检索、BM25、混合召回、重排和引用溯源。
- 理解 Chroma、Qdrant、Milvus 在不同规模和场景下的取舍。
- 对照 LightRAG 与 RAGFlow，理解知识图谱 RAG 和企业文档 RAG。
- 建立可量化的检索评估方法，而不是仅凭回答观感调参。

## 学习主链路

```text
原始文档
  → 文档解析
  → Chunk 分块
  → Embedding
  → 向量存储
  → Metadata Filter
  → TopK 召回
  → 混合检索
  → Reranker
  → LLM 生成
  → 引用溯源
```

学习每个模块时都应回答：

- 输入和输出是什么？
- 数据与状态保存在哪里？
- 精度、延迟、内存和成本如何权衡？
- 失败或结果不相关时，问题发生在哪一层？
- 如何用指标和样例验证改动确实有效？

## 实施原则

1. 先实现最小可运行闭环，再逐步增加数据库和框架。
2. 先手写余弦相似度、TopK 和 Metadata Filter，再使用 Chroma 等产品。
3. 使用统一 `VectorStore` 抽象接入 Memory、Chroma、Qdrant 和 Milvus，避免业务代码绑定具体产品。
4. Milvus 重点学习索引实验和分布式架构，不以从零复刻整个系统为目标。
5. LightRAG 和 RAGFlow 放在完整检索链路之后学习，避免只会配置平台而不理解底层检索。
6. 每个阶段都准备固定问题和预期结果，记录召回质量与性能数据。
7. API Key 和连接凭据只通过环境变量或本地忽略文件提供，禁止提交。

## 阶段路线

- [ ] 阶段 0：手写内存向量检索
- [ ] 阶段 1：Chroma 与基础知识库
- [ ] 阶段 2：Qdrant 与生产级过滤检索
- [ ] 阶段 3：Milvus 索引实验与分布式架构
- [ ] 阶段 4：完整 RAG 与引用溯源
- [ ] 阶段 5：BM25、混合检索、重排与评估
- [ ] 阶段 6：LightRAG / RAGFlow 对照实验

详细任务和验收标准见 [实施计划](./docs/IMPLEMENTATION_PLAN.md)。

## 技术基线

- Python 3.12+
- `uv` 管理 Python、虚拟环境、依赖和锁文件
- `src/lob_vector` 包结构
- `pyproject.toml` 管理构建、安装和 CLI 入口
- 阶段 0 优先使用 Python 标准库，确有向量计算需要时再引入 NumPy

当前已完成基础工程骨架、核心数据模型、固定长度分块，以及用于本地验证链路的确定性 Hash Embedding。

初始化环境：

```bash
uv sync
```

运行 CLI：

```bash
uv run lob-vector --help
uv run lob-vector --version
uv run lob-vector chunk ./README.md --chunk-size 500 --overlap 50
uv run lob-vector embed "什么是向量检索" --dimension 32
uv run lob-vector search "向量检索是什么" README.md docs/IMPLEMENTATION_PLAN.md
```

查看附带 Metadata 的分块结果：

```bash
uv run lob-vector chunk ./README.md \
  --chunk-size 500 \
  --overlap 50 \
  --metadata department=engineering \
  --metadata year=2026
```

启动可视化分块实验台：

```bash
uv run lob-vector web
```

浏览器打开 <http://127.0.0.1:8765>，可以粘贴文本并调整 Chunk Size、Overlap、向量维度和 Metadata，也可以输入问题体验 TopK 检索与 Metadata Filter。页面结果直接来自 Python `FixedSizeChunker`、`HashEmbedder` 与 `MemoryVectorStore`。

`HashEmbedder` 不具备真实模型的语义理解能力，只用于无网络环境下验证 Embedding、向量归一化和后续检索链路；真实 Embedding 模型将在接口稳定后接入。

## 第一个里程碑

导入 10 篇本地文档，手写余弦相似度与 Metadata Filter。输入问题后，返回最相关的 3 个 Chunk，并展示：

- 文本内容
- 相似度分数
- 文件名
- 页码或段落位置
- Metadata 过滤条件

这个闭环完成后，再将内存实现替换为 Chroma，并保持上层查询代码不变。

## 推荐模块结构

```text
lob-vector/
├── cmd/
│   └── lob-vector/
├── internal/
│   ├── document/
│   ├── parser/
│   ├── chunker/
│   ├── embedding/
│   ├── vectorstore/
│   │   ├── memory/
│   │   ├── chroma/
│   │   ├── qdrant/
│   │   └── milvus/
│   ├── retrieval/
│   ├── reranker/
│   ├── generation/
│   └── evaluation/
├── examples/
├── datasets/
└── docs/
```

阶段 0 使用 Python 实现，并在形成最小闭环前保持依赖精简。

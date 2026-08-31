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

- [x] 阶段 0：手写内存向量检索
- [x] 阶段 1：Chroma 与基础知识库
- [x] 阶段 2：Qdrant 与生产级过滤检索
- [x] 阶段 3：Milvus 索引实验与分布式架构
- [x] 阶段 4：完整 RAG 与引用溯源
- [x] 阶段 5：BM25、混合检索、重排与评估
- [x] 阶段 6：LightRAG 图检索实验
- [x] 阶段 7：RAGFlow 平台能力与最终选型

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

`HashEmbedder` 不具备真实模型的语义理解能力，只用于无网络环境下验证 Embedding、向量归一化和后续检索链路；需要语义检索时使用百炼 Embedding。

使用阿里云百炼真实 Embedding（API Key 只从环境变量读取）：

在项目根目录 `.env` 中配置（该文件已被 Git 忽略）：

```dotenv
DASHSCOPE_API_KEY=你的百炼_API_Key
```

然后执行：

```bash
uv run lob-vector embed "什么是向量检索" --embedder bailian
uv run lob-vector search "RAG 检索需要掌握什么" \
  datasets/knowledge-base/*.md \
  --embedder bailian
```

默认模型为 `text-embedding-v4`，向量维度为 1024。若使用百炼业务空间专属域名，追加
`--base-url https://你的业务空间ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。

使用 Chroma 建立并查询持久化知识库：

```bash
uv run lob-vector index datasets/demo-knowledge-base/*.md \
  --store chroma \
  --collection demo-kb

# 新进程中不再传文件，直接查询磁盘中的 Collection
uv run lob-vector search "登录凭据想不起来怎么办" \
  --store chroma \
  --collection demo-kb

uv run lob-vector clear --collection demo-kb
```

Chroma 默认持久化到 `.chroma/`，该目录已被 Git 忽略。Memory 与 Chroma 都实现统一
`VectorStore` 的 `upsert`、`delete`、`clear`、`count` 和 `search` 能力。

使用 Qdrant 本地模式建立并查询持久化知识库：

```bash
uv run lob-vector index datasets/demo-knowledge-base/*.md \
  --store qdrant \
  --collection demo-kb \
  --metadata tenant_id=tenant-a \
  --metadata department=support \
  --metadata permission=staff

uv run lob-vector search "登录凭据想不起来怎么办" \
  --store qdrant \
  --collection demo-kb \
  --where tenant_id=tenant-a \
  --where department=support \
  --where permission=staff

uv run lob-vector clear --store qdrant --collection demo-kb
```

Qdrant 默认持久化到 `.qdrant/`，支持与 Memory、Chroma 相同的 Metadata AND 组合过滤。

生产 Server 模式下，Stage 2 网页还提供可复现的 HNSW 实验：固定随机种子生成向量，
以 `exact=true` 的 Top 10 作为 Ground Truth，对比 `hnsw_ef=16/64/128` 的 Recall@10、
平均延迟、P50 和 P95，并观察删除、恢复前后的 Point、索引与 Segment 状态。

### 使用 Qdrant Server

`.env` 使用 `QDRANT_MODE` 显式切换模式。默认体验级配置无需启动容器，也不需要密钥：

```dotenv
QDRANT_MODE=local
```

此时数据保存在 `.qdrant/`，适合单进程学习。切换生产 Server 时必须同时配置 URL 和密钥：

```dotenv
QDRANT_MODE=server
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=替换成高强度随机密钥
```

`server` 模式缺少 URL 或密钥时，应用会直接拒绝连接。使用项目提供的 Docker Compose 启动服务：

```bash
cp .env.example .env
# 将 QDRANT_API_KEY 替换为高强度随机值
docker compose up -d qdrant
docker compose ps
```

CLI 与 Web 实验台读取同一份 `.env`，因此切换 `QDRANT_MODE` 后重启应用即可切换后端。

```bash
uv run lob-vector index datasets/demo-knowledge-base/*.md \
  --store qdrant \
  --collection demo-kb \
  --metadata tenant_id=tenant-a \
  --metadata department=support \
  --metadata permission=staff

uv run lob-vector web
```

服务仅绑定本机 `127.0.0.1`，启用 API Key、健康检查、自动重启、资源限制、持久化数据卷和
快照卷。需要跨机器访问时，应放在 TLS 反向代理或私有网络之后，不要直接把 6333/6334
暴露到公网。单节点 Compose 提供生产化运行基线，但不等同于高可用集群。

### 使用 Milvus

默认使用 Milvus Lite，不需要 Docker，数据保存在 `.milvus/lob-vector.db`：

```dotenv
MILVUS_MODE=lite
```

```bash
uv run lob-vector index datasets/demo-knowledge-base/*.md \
  --store milvus \
  --collection demo_milvus

uv run lob-vector search "登录凭据想不起来怎么办" \
  --store milvus \
  --collection demo_milvus
```

Lite 适合验证统一 `VectorStore`、持久化和 Metadata Filter。运行 FLAT、IVF_FLAT、HNSW
索引对照时，切换到 Milvus Standalone：

```dotenv
MILVUS_MODE=server
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=
```

```bash
docker compose --profile milvus up -d milvus
docker compose --profile milvus ps
uv run lob-vector web
```

Compose 会启动 Milvus 2.6 Standalone、etcd 和 MinIO。Milvus WebUI 位于
<http://127.0.0.1:9091/webui/>。Stage 3 网页使用固定随机种子和相同查询，对照：

- FLAT：全量扫描，作为 Recall@10 的精确基准。
- IVF_FLAT：`nlist=128`、`nprobe=16`，观察先分桶后扫描的精度与延迟。
- HNSW：`M=16`、`efConstruction=100`、`ef=64`，观察图索引的内存换延迟路径。

页面同时展示 Proxy、Coordinator、Query Node、Data Node、etcd 和 MinIO 在查询与索引链路中的职责。

### 运行完整 RAG 与引用实验

Stage 4 使用百炼 Embedding 检索典型知识库，再由 `qwen-plus` 严格基于证据生成回答：

```dotenv
DASHSCOPE_API_KEY=你的百炼_API_Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen-plus
```

```bash
uv run lob-vector web
```

进入 Stage 4 后点击“检索并生成”。页面会在发送知识库内容到百炼前要求确认，并展示：

- TopK 候选数量与最低证据分数。
- 最终进入 Prompt 的证据、来源文件、章节和字符位置。
- 回答中的 `[1]` 引用与对应原文。
- 检索、生成耗时和 Token 用量。
- 高阈值或知识库外问题的拒答结果。

网页默认通过统一 `VectorStore` 使用 Qdrant 完成检索；`/api/rag-answer` 也支持将 `store`
设置为 `memory`、`chroma`、`qdrant` 或 `milvus`，用于验证生成层不依赖具体向量数据库。

程序只接受指向当前证据编号的引用；模型未给出有效引用时，即使生成了文本也会降级为“根据当前知识库资料，无法回答这个问题”。

### 运行混合检索与评估

Stage 5 使用 6 个带预期来源和章节的固定问题，对照四种策略：

- 向量检索：Hash 离线基线或百炼真实语义 Embedding。
- BM25：针对关键词、错误码、型号和专有名词的稀疏检索。
- RRF：仅使用名次融合向量与 BM25，避免直接比较不同分数空间。
- 透明 Reranker：根据问题词覆盖、章节命中和原排名重新排序。

页面同时输出单题 Top 3，以及固定评估集的 Recall@3、MRR、NDCG 和平均检索耗时。
混合或重排没有超过最佳单路结果时，页面会保留真实指标，不预设“策略越复杂一定越好”。

### 运行 LightRAG 图检索对照

Stage 6 通过独立 LightRAG Server 对照四种查询模式：`naive` 只查文本 Chunk，`local` 查具体实体及邻接关系，`global` 查跨文档关系，`mix` 合并图检索与文本检索。

先在 `.env` 中设置 `DASHSCOPE_API_KEY`、`LIGHTRAG_API_KEY`，并将 `LIGHTRAG_MODE` 改为 `server`，然后启动：

```bash
docker compose --profile lightrag up -d lightrag
uv run lob-vector web
```

打开 `http://127.0.0.1:8765/#chapter-6`，先点击“索引典型知识库”。实体关系抽取是异步模型任务，可在 `http://127.0.0.1:9621/webui` 查看进度；处理完成后再运行四模式对照。

这一实验会真实调用百炼：建图阶段需要逐段抽取实体与关系，查询阶段每种模式也会调用模型。页面在外发资料或问题前都会要求确认。LightRAG 端口只绑定本机，并要求 `X-API-Key`，不要把空密钥或控制台直接暴露到公网。

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

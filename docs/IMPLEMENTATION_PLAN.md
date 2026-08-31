# LOB Vector 实施计划

## 1. 要掌握的系统是什么

向量数据库只是 RAG 链路中的一层。完整系统需要将原始文档转换成可检索数据，根据用户问题召回相关证据，经过融合与重排后交给大模型，并把答案映射回可靠的原文来源。

```text
PDF / Word / Markdown / Web
              ↓
       Parser → Chunker
              ↓
          Embedding
              ↓
      VectorStore + Metadata
              ↓
Vector Recall + BM25 Recall
              ↓
       Fusion → Reranker
              ↓
        Prompt → LLM
              ↓
      Answer + Citations
```

本项目先实现纵向最小闭环，再将每一层替换成真实产品进行对照。重点不是记忆 SDK，而是理解检索结果为什么正确、为什么遗漏，以及规模增长后系统如何变化。

## 2. 产品学习定位

| 产品 | 重点概念 | 在本项目中的作用 | 典型场景 |
|---|---|---|---|
| Chroma | Collection、Embedding、Metadata Filter、持久化 | 第一个外部向量存储实现 | 本地知识库、原型和小型应用 |
| Qdrant | Segment、Payload Index、HNSW、过滤检索 | 生产级过滤与在线检索实验 | 多租户、权限过滤、推荐系统 |
| Milvus | Partition、Shard、索引构建、Query Node | 大规模索引与分布式实验 | 海量向量、高并发检索中台 |
| LightRAG | 实体、关系、图检索、全局与局部查询 | 知识图谱 RAG 对照实现 | 跨文档关系和全局总结 |
| RAGFlow | 文档解析、混合检索、重排、引用 | 企业文档 RAG 对照平台 | PDF、制度、合同和产品手册 |

## 3. 分阶段实施

### 阶段 0：手写内存向量检索

目标：不依赖向量数据库，打通最小查询链路。

任务：

1. 定义 `Document`、`Chunk`、`Metadata` 和查询结果结构。
2. 实现固定长度分块，并保留文件名、页码或段落位置。
3. 定义 `Embedder` 接口，提供真实模型和可重复的本地假实现。
4. 实现向量归一化、余弦相似度、全量扫描和 TopK。
5. 实现等值、范围和组合 Metadata Filter。
6. 提供 CLI：导入文档、执行查询并展示分数与来源。

验收：导入至少 10 篇文档，查询可以返回 Top 3 Chunk、相似度、文件名和位置；过滤条件能排除语义相似但不符合范围的数据。

必须理解：

- 余弦相似度与点积的关系。
- Embedding 维度和归一化的影响。
- TopK 排序和过滤顺序。
- Chunk 大小与 overlap 对召回的影响。

### 阶段 1：Chroma 与基础知识库

目标：将内存存储替换为 Chroma，上层检索流程保持不变。

任务：

1. 定义统一的 `VectorStore` 接口。
2. 实现 Memory 和 Chroma 两种适配器。
3. 映射 Collection、ID、Document、Embedding 和 Metadata。
4. 支持批量写入、更新、删除和相似度查询。
5. 支持 Metadata Filter 与本地持久化。
6. 验证程序重启后数据和查询结果可恢复。

验收：同一组数据和问题可以在 Memory 与 Chroma 间切换；业务层不需要修改；关键查询结果基本一致。

### 阶段 2：Qdrant 与生产级过滤检索

目标：掌握向量检索与复杂结构化过滤的协同方式。

任务：

1. 实现 Qdrant `VectorStore` 适配器。
2. 映射 Collection、Point、Vector 与 Payload。
3. 为租户、部门、时间和权限字段建立 Payload Index。
4. 实现组合 Filter 与向量 TopK 查询。
5. 调整 HNSW 构建和查询参数，记录延迟与召回变化。
6. 观察 Segment、数据更新、删除和持久化行为。

实验场景：

```text
tenant_id = 当前租户
AND department = 当前部门
AND permission 包含当前用户角色
AND 文档内容与问题语义相似
```

验收：构造多租户数据集，确认任何查询都不会召回无权限数据；对 HNSW 至少进行一组可复现的参数对比。

### 阶段 3：Milvus 索引实验与分布式架构

目标：理解向量数据规模增加后，分片、索引和查询节点如何协作。

任务：

1. 部署 Milvus，并完成 Collection Schema 与数据导入。
2. 使用 Partition 或业务字段组织数据。
3. 对比 FLAT、HNSW 和一种 IVF 索引。
4. 生成 10 万和 100 万级测试向量。
5. 测量索引构建时间、查询延迟、召回率和资源占用。
6. 梳理代理、协调组件、Query Node、Data Node、存储之间的调用关系。
7. 记录写入、Flush、数据加载与查询可见性的生命周期。

验收：形成一份索引对比记录，能够解释不同索引在当前数据规模下的取舍，并画出一次查询经过的分布式调用链。

注意：本阶段不要求从零复刻 Milvus，重点是实验和理解架构。

### 阶段 4：完整 RAG 与引用溯源

目标：基于检索结果生成有证据、可追溯的答案。

任务：

1. 支持 Markdown、PDF，并按需要扩展 Word 解析。
2. 实现固定长度与递归分块策略。
3. 保留文档 ID、版本、页码、标题层级和 Chunk ID。
4. 将 TopK 证据组织成结构化 Prompt。
5. 接入生成模型并限制其仅依据检索证据回答。
6. 输出答案、引用编号和对应原文片段。
7. 证据不足时明确拒答或提示信息不足。

验收：准备一组可核对答案的问题；每个有效回答都能定位到原始文件和具体位置；无依据问题不会被当作有依据答案输出。

### 阶段 5：混合检索、重排与评估

目标：解决型号、错误码、专有名词和语义问题混合出现时的召回质量。

任务：

1. 实现 BM25 关键词检索。
2. 并行执行 BM25 与向量召回。
3. 使用 RRF 等方法融合两路结果。
4. 接入 Cross-Encoder 或同类 Reranker。
5. 建立 30～50 个问题的最小评估集，标注相关文档或 Chunk。
6. 统计 Recall@K、MRR、NDCG 和端到端延迟。
7. 对比纯向量、纯 BM25、混合检索和重排后的结果。

验收：评估脚本能重复运行并输出指标；能够用数据说明某项策略是否改善召回，而不是只依靠主观观察。

### 阶段 6：LightRAG 图检索实验

目标：理解知识图谱 RAG 如何增加实体关系与跨文档查询能力。

LightRAG 任务：

1. [x] 使用与自研链路相同的一批文档。
2. [x] 通过独立控制台观察实体与关系抽取结果。
3. [x] 对比 naive、local、global 和 mix，并展示回答、引用与耗时。
4. [x] 提供跨文档关系与全局总结问题作为固定实验入口。

### 阶段 7：RAGFlow 平台能力与最终选型

本项目不安装 RAGFlow 完整栈，重点理解以下产品能力：

1. [x] 说明 PDF、Markdown、Word 等文档导入与解析能力。
2. [x] 说明切片查看、版面保留和失败任务管理能力。
3. [x] 说明关键词、向量、混合召回和重排调试能力。
4. [x] 说明答案引用、来源文件和原文定位能力。
5. [x] 对照自研 RAG、LightRAG 与 RAGFlow 的边界和适用场景。

验收：页面将 LightRAG 与 RAGFlow 拆成独立阶段，并能说明三种方案各自更适合的问题类型及工程场景。

## 4. 推荐模块边界

```text
cmd/lob-vector        CLI 或服务入口
internal/document     文档、Chunk 和来源模型
internal/parser       文件解析与版面信息提取
internal/chunker      分块策略
internal/embedding    Embedding 接口与供应商适配
internal/vectorstore  Memory、Chroma、Qdrant、Milvus
internal/retrieval    向量、BM25、融合与过滤
internal/reranker     精排模型适配
internal/generation   Prompt 与回答生成
internal/evaluation   数据集、指标与对比实验
```

依赖方向保持单向：入口负责编排，核心数据模型不依赖具体数据库 SDK，产品适配器实现核心接口。

## 5. 建议保留的实验记录

每个阶段至少记录：

- 数据集规模和来源。
- Embedding 模型、维度与距离度量。
- Chunk 大小、overlap 和 Metadata 字段。
- 索引类型及关键参数。
- Recall@K、MRR 或人工相关性判断。
- P50、P95 查询延迟和资源占用。
- 错误案例、原因分析和下一步调整。

## 6. 学习完成标准

完成本项目后，应能独立回答并通过实验验证：

1. 小型知识库为什么可以选择 Chroma。
2. 多租户和复杂权限过滤为什么更适合 Qdrant 一类生产服务。
3. 数据达到大规模后为什么需要 Milvus 的分片与分布式查询能力。
4. BM25、向量召回和 Reranker 分别解决什么问题。
5. LightRAG 的图检索相对普通 RAG 增加了什么能力。
6. RAGFlow 在底层向量数据库之上提供了哪些文档和产品能力。
7. 如何通过评估集和指标判断一次 RAG 优化是否真正有效。

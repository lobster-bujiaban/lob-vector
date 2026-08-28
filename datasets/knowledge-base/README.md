# 共享知识库检索数据集

用于阶段 0 内存向量检索的真实中文语料，来源于本地共享知识库。

| 文件 | 内容 | 建议 Metadata |
|---|---|---|
| `tech-stack.md` | 技术栈、领域经验与默认工程选型 | `category=profile`、`source=shared-knowledge-base` |
| `ai-application-engineer-roadmap.md` | AI 应用工程师 L0～L6 学习路径 | `category=learning`、`source=shared-knowledge-base` |

原始资料仍以共享知识库为准，本目录仅保存实验副本。资料更新后需要重新同步，避免使用过期内容。

本地检索示例：

```bash
uv run lob-vector search "RAG 检索需要掌握什么" \
  datasets/knowledge-base/tech-stack.md \
  datasets/knowledge-base/ai-application-engineer-roadmap.md
```

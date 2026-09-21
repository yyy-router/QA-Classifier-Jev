# QA-Classifier-Jev

用 TypeSafe AI 的 Jev/System One API 复现实验一个问题分类器。当前目标是把历史 SetFit 分类器的数据和标签体系保留下来，用 Jev 的 `choice` 问题做三分类对比。

## Git 工作流

- 主分支只接受 PR 合并。
- 每个实现点在独立分支开发，例如 `chore/init-project-structure`、`feature/jev-eval`。
- 不直接 push 代码；push 前先确认。
- `.env`、API key、历史模型文件、实验输出不进入 Git。

## Conda 环境

```powershell
conda env create -f environment.yml
conda activate qa-classifier-jev
pip install -e .
```

配置 API key：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填入：

```text
TYPESAFE_API_KEY=your_api_key_here
```

## 数据来源

历史分类器和数据暂存在 `history/`，该目录默认被 `.gitignore` 忽略，不会提交到仓库。默认评测集路径：

```text
history/data/test.json
```

数据格式：

```json
[
  {"text": "What is photosynthesis?", "label": "Definition"}
]
```

## 标签体系

标签定义在 `config/label_schema_v1.json`：

- `Fact`
- `Definition`
- `Reason`

Jev 调用时会把这些标签作为 `choice.criteria` 传入，返回预测标签、概率分布和置信度。

## 本地检查

无 API key 时可以生成请求 payload：

```powershell
qa-jev-classify --question "Why is the sky blue?" --dry-run
```

有 API key 后进行单条分类：

```powershell
qa-jev-classify --question "Why is the sky blue?"
```

跑固定测试集：

```powershell
qa-jev-evaluate --dataset history/data/test.json --limit 20
```

完整评测会写入：

```text
outputs/jev_predictions.jsonl
outputs/metrics.json
```

## 官方接口参考

- TypeSafe Quick start: https://docs.typesafe.ai/introduction/quickstart
- HTTP API reference: https://docs.typesafe.ai/api

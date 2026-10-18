# AI Agent 学习笔记

本仓库用于记录我在 **AI Agent 方向** 的学习过程与代码实践，每个项目一个独立文件夹，持续更新。

## 项目列表

| 目录 | 内容 | 技术栈 |
|------|------|--------|
| [01-Agent_react](./01-Agent_react) | 从零实现最小 ReAct Agent：Thought → Action → Observation 循环，内置计算器 / 时间 / 字数统计三个工具 | TypeScript、OpenRouter（OpenAI 兼容接口） |

每个项目目录内有自己的 `Readme.md`（运行方式）和 `logic.txt`（代码逻辑图）。

## 运行方式

以 `01-Agent_react` 为例：

```bash
cd 01-Agent_react
npm install
cp .env.example .env   # 填入 OPENROUTER_API_KEY
npm start
```

密钥只放在本地 `.env`，已被 `.gitignore` 排除，不会提交。

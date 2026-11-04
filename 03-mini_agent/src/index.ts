/**
 * mini-agent —— 入口文件
 *
 * 职责只有装配：加载 .env → 读配置 → 发一次请求 → 打印结果。
 * 这里不放任何需要被测试的业务逻辑，因此不需要入口守卫。
 */

import { loadConfig } from "./config.js";
import { chat } from "./llm.js";

/**
 * 加载项目根目录的 .env。
 * 文件不存在时静默忽略 —— 允许直接用 shell 环境变量（CI 场景常见）。
 */
function loadDotEnv(): void {
  try {
    process.loadEnvFile(".env");
  } catch {
    // 没有 .env 不是错误
  }
}

async function main(): Promise<void> {
  loadDotEnv();

  const config = loadConfig();

  console.log(`模型: ${config.model}`);
  console.log(`端点: ${config.baseUrl}`);
  console.log("─────────────────────────────");

  const result = await chat(config, [
    {
      role: "system",
      content: "你是一名严谨的投资研究助理。回答简洁，只陈述事实，不做投资建议。",
    },
    {
      role: "user",
      content: "用一句话说明毛利率反映什么。",
    },
  ]);

  console.log(result.text);

  if (result.usage) {
    console.log("─────────────────────────────");
    console.log(
      `tokens: 输入 ${result.usage.promptTokens} / 输出 ${result.usage.completionTokens} / 合计 ${result.usage.totalTokens}`,
    );
  }
}

main().catch((error: unknown) => {
  // 配置错误和调用错误都是预期内的失败，打印信息即可，不需要堆栈
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

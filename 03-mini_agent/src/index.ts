/**
 * mini-agent —— 入口文件
 *
 * 当前里程碑（Week 1 / Task 01-02）：
 *   1. 验证 TypeScript 工程链路（编译、类型检查、运行）
 *   2. 验证消息契约（Zod schema + 类型收窄）
 *
 * 刻意不包含：LLM API 调用、Agent Loop。
 * 这些会在后续任务里逐个引入，每一步都保持可运行、可验证。
 */

// 注意后缀是 .js 而不是 .ts —— 这是 Node ESM 的解析规则：
// 运行时导入的是编译后的 dist/types.js，TypeScript 源码后缀不出现在导入路径里。
import { messageSchema, type Message } from "./types.js";

/** 运行时的基本信息。用 readonly 表达"这个对象构造后不应被修改"。 */
interface RuntimeInfo {
  readonly nodeVersion: string;
  readonly platform: NodeJS.Platform;
  readonly arch: string;
  readonly cwd: string;
}

function getRuntimeInfo(): RuntimeInfo {
  return {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cwd: process.cwd(),
  };
}

function printRuntimeInfo(): void {
  const info = getRuntimeInfo();

  console.log("mini-agent · 环境自检");
  console.log("─────────────────────────────");
  console.log(`Node.js  : ${info.nodeVersion}`);
  console.log(`平台     : ${info.platform} / ${info.arch}`);
  console.log(`工作目录 : ${info.cwd}`);
}

/**
 * 描述一条消息，用于展示判别联合的类型收窄。
 * 每个分支里 msg 都是具体变体，能访问该变体独有的字段。
 */
function describeMessage(msg: Message): string {
  switch (msg.role) {
    case "system":
      return `[system] 系统提示词，${msg.content.length} 字符`;
    case "user":
      return `[user] ${msg.content}`;
    case "assistant": {
      const toolPart =
        msg.toolCalls && msg.toolCalls.length > 0
          ? `，请求调用 ${msg.toolCalls.map((c) => c.name).join(", ")}`
          : "";
      return `[assistant] ${msg.content || "(无文本)"}${toolPart}`;
    }
    case "tool":
      // 只有走到这个分支，msg.toolCallId 才是合法字段。
      return `[tool] ${msg.name} 返回结果，关联调用 ${msg.toolCallId}`;
  }
}

function printContractCheck(): void {
  console.log("─────────────────────────────");
  console.log("消息契约自检");

  // 1) 合法输入：应通过校验，并被推导为 Message 类型
  const valid = messageSchema.safeParse({
    role: "user",
    content: "帮我看看贵州茅台 2024 年报的毛利率变化",
  });

  if (!valid.success) {
    console.log("✗ 预期通过，实际被拒绝：", valid.error.issues);
    return;
  }
  console.log(`✓ 合法消息通过：${describeMessage(valid.data)}`);

  // 2) 非法输入：user 消息 content 为空，应被拒绝
  const invalid = messageSchema.safeParse({ role: "user", content: "" });

  if (invalid.success) {
    console.log("✗ 预期被拒绝，实际通过了 —— 契约存在漏洞");
    return;
  }
  const first = invalid.error.issues[0];
  console.log(
    `✓ 非法消息被拒绝：${first?.path.join(".") || "(root)"} — ${first?.message ?? "unknown"}`,
  );

  // 3) 带工具调用的助手消息：验证可选字段与嵌套结构
  const withToolCall = messageSchema.safeParse({
    role: "assistant",
    content: "",
    toolCalls: [
      {
        id: "call_001",
        name: "get_financial_report",
        arguments: { ticker: "600519", year: 2024 },
      },
    ],
  });

  if (!withToolCall.success) {
    console.log("✗ 带工具调用的消息被拒绝：", withToolCall.error.issues);
    return;
  }
  console.log(`✓ 工具调用消息通过：${describeMessage(withToolCall.data)}`);
}

function main(): void {
  printRuntimeInfo();
  printContractCheck();
  console.log("─────────────────────────────");
  console.log("TypeScript 工程链路与消息契约均正常。");
}

main();

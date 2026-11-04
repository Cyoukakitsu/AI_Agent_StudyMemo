/**
 * 最小 LLM 调用封装（OpenAI 兼容的 /chat/completions 协议）。
 *
 * 为什么不用官方 SDK：本次目标是看清协议本身 —— 请求长什么样、响应怎么校验。
 * 一个 fetch 调用能把这件事暴露得最彻底；后续若需要流式、重试、并发控制，再评估是否引入。
 *
 * 兼容范围：任何实现 OpenAI Chat Completions 协议的服务
 * （DeepSeek、OpenAI、月之暗面、智谱、通义等），只需改 LLM_BASE_URL 与 LLM_MODEL。
 */

import { z } from "zod";

import type { Config } from "./config.js";

/** 发往模型的消息。这里只覆盖纯文本，工具调用留到后续任务。 */
export interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}

/**
 * 响应结构校验。
 *
 * 只声明我们真正会读取的字段。模型返回的其他字段会被 Zod 默认剥离，
 * 这样上游随意新增字段不会污染内部状态。
 */
const completionSchema = z.object({
  choices: z
    .array(
      z.object({
        message: z.object({
          // 部分服务会返回非 "assistant" 的值，这里不强约束，交由业务判断
          role: z.string(),
          // 只返回工具调用时 content 可能为 null
          content: z.string().nullable(),
        }),
        finish_reason: z.string().nullable().optional(),
      }),
    )
    .min(1),
  usage: z
    .object({
      prompt_tokens: z.number().int().nonnegative(),
      completion_tokens: z.number().int().nonnegative(),
      total_tokens: z.number().int().nonnegative(),
    })
    .optional(),
});

export interface TokenUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
}

export interface ChatResult {
  readonly text: string;
  readonly finishReason: string | null;
  readonly usage?: TokenUsage;
}

/** 调用失败单独成类，区分于配置错误和编程错误。 */
export class LlmError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "LlmError";
  }
}

const TIMEOUT_MS = 60_000;

/** 截断长文本，避免把整个响应体打进错误信息。 */
function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}…`;
}

/**
 * 发起一次对话请求，返回模型的文本回复。
 *
 * 三层校验，缺一不可：
 *   1. HTTP 状态码 —— 认证失败、限流、服务端错误
 *   2. JSON 可解析 —— 网关有时会返回 HTML 错误页
 *   3. 结构符合预期 —— 字段缺失或类型不符
 */
export async function chat(
  config: Config,
  messages: readonly ChatMessage[],
): Promise<ChatResult> {
  const response = await fetch(`${config.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      messages,
      stream: false,
    }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });

  // 先取文本再解析：失败时能带上原始响应片段，便于排查
  const raw = await response.text();

  if (!response.ok) {
    throw new LlmError(
      `请求失败 HTTP ${response.status}：${truncate(raw, 300)}`,
      response.status,
    );
  }

  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch {
    throw new LlmError(`响应不是合法 JSON：${truncate(raw, 300)}`);
  }

  const parsed = completionSchema.safeParse(json);
  if (!parsed.success) {
    const detail = parsed.error.issues
      .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`)
      .join("; ");
    throw new LlmError(`响应结构不符合预期：${detail}`);
  }

  const choice = parsed.data.choices[0];
  if (!choice) {
    throw new LlmError("响应中 choices 为空");
  }

  const text = choice.message.content ?? "";
  if (text.trim() === "") {
    throw new LlmError("模型返回了空内容");
  }

  const usage = parsed.data.usage;

  return {
    text,
    finishReason: choice.finish_reason ?? null,
    ...(usage
      ? {
          usage: {
            promptTokens: usage.prompt_tokens,
            completionTokens: usage.completion_tokens,
            totalTokens: usage.total_tokens,
          },
        }
      : {}),
  };
}

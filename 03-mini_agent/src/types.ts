/**
 * Agent 的核心数据契约。
 *
 * 设计原则：
 * 1. Schema 是唯一真相 —— TypeScript 类型由 z.infer 推导，避免"类型和校验逻辑两处维护"。
 * 2. 用判别联合（discriminated union）而非扁平接口 —— 让 `switch (msg.role)` 能自动收窄类型，
 *    并保证"tool 消息必然带 toolCallId"这类约束由类型系统强制，而非靠注释约定。
 * 3. 校验只发生在边界（LLM 响应、用户输入、数据库读取），内部流转不重复校验。
 */

import { z } from "zod";

/* ------------------------------------------------------------------ *
 * Role
 * ------------------------------------------------------------------ */

/**
 * 消息的生产者。
 * - system    : 系统提示词，设定 Agent 的行为边界
 * - user      : 人类输入
 * - assistant : 模型输出（可能带工具调用请求）
 * - tool      : 工具执行结果，回灌给模型
 */
export const roleSchema = z.enum(["system", "user", "assistant", "tool"]);
export type Role = z.infer<typeof roleSchema>;

/* ------------------------------------------------------------------ *
 * 工具调用
 * ------------------------------------------------------------------ */

/**
 * 模型发起的一次工具调用请求。
 * arguments 用 record 而非具体结构：不同工具的入参形状不同，
 * 具体校验由各工具自己的 schema 负责，这里只保证"是一个对象"。
 */
export const toolCallSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  arguments: z.record(z.string(), z.unknown()),
});
export type ToolCall = z.infer<typeof toolCallSchema>;

/* ------------------------------------------------------------------ *
 * Message（按 role 区分的四个变体）
 * ------------------------------------------------------------------ */

/** 系统提示词允许为空字符串，便于先占位后填充。 */
const systemMessageSchema = z.object({
  role: z.literal("system"),
  content: z.string(),
});

/** 用户消息不允许为空 —— 空输入通常意味着上游有 bug，尽早暴露。 */
const userMessageSchema = z.object({
  role: z.literal("user"),
  content: z.string().min(1),
});

/**
 * 助手消息允许 content 为空：模型可能只返回工具调用而不附带任何文本。
 * toolCalls 可选 —— 绝大多数回复不带工具调用。
 */
const assistantMessageSchema = z.object({
  role: z.literal("assistant"),
  content: z.string(),
  toolCalls: z.array(toolCallSchema).optional(),
});

/** 工具结果通过 toolCallId 与发起的调用配对。 */
const toolMessageSchema = z.object({
  role: z.literal("tool"),
  toolCallId: z.string().min(1),
  name: z.string().min(1),
  content: z.string(),
});

export const messageSchema = z.discriminatedUnion("role", [
  systemMessageSchema,
  userMessageSchema,
  assistantMessageSchema,
  toolMessageSchema,
]);

export type Message = z.infer<typeof messageSchema>;

/** 各变体的具名类型，便于函数签名精确表达"只接受用户消息"。 */
export type SystemMessage = z.infer<typeof systemMessageSchema>;
export type UserMessage = z.infer<typeof userMessageSchema>;
export type AssistantMessage = z.infer<typeof assistantMessageSchema>;
export type ToolMessage = z.infer<typeof toolMessageSchema>;

/* ------------------------------------------------------------------ *
 * Conversation
 * ------------------------------------------------------------------ */

/**
 * 一次完整对话。至少包含一条消息 —— 空对话在 Agent 语义下没有意义。
 * 后续接入持久化时，这会直接对应一张表的行集合。
 */
export const conversationSchema = z.array(messageSchema).min(1);
export type Conversation = z.infer<typeof conversationSchema>;

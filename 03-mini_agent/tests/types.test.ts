/**
 * 消息契约的单元测试。
 *
 * 测试重点不是"正常输入能通过"（那是显然的），而是边界行为：
 * 哪些输入必须被拒绝、可选字段缺省后是什么形态、多余字段如何处理。
 * 这些恰恰是后续重构中最容易被悄悄改坏的部分。
 */

import { describe, expect, it } from "vitest";

import { conversationSchema, messageSchema, roleSchema } from "../src/types.js";

describe("roleSchema", () => {
  it("接受四种合法角色", () => {
    for (const role of ["system", "user", "assistant", "tool"]) {
      expect(roleSchema.safeParse(role).success).toBe(true);
    }
  });

  it("拒绝未知角色", () => {
    expect(roleSchema.safeParse("developer").success).toBe(false);
  });
});

describe("messageSchema · user", () => {
  it("接受非空 content", () => {
    const result = messageSchema.safeParse({
      role: "user",
      content: "分析一下 600519 的估值",
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.role).toBe("user");
    }
  });

  it("拒绝空 content", () => {
    const result = messageSchema.safeParse({ role: "user", content: "" });

    expect(result.success).toBe(false);
    if (!result.success) {
      // 错误必须定位到 content 字段，而不是笼统的根级错误
      expect(result.error.issues[0]?.path).toEqual(["content"]);
    }
  });
});

describe("messageSchema · system", () => {
  it("允许空 content，便于先占位后填充", () => {
    const result = messageSchema.safeParse({ role: "system", content: "" });

    expect(result.success).toBe(true);
  });
});

describe("messageSchema · assistant", () => {
  it("不带 toolCalls 时，toolCalls 为 undefined 而非空数组", () => {
    const result = messageSchema.safeParse({
      role: "assistant",
      content: "毛利率同比提升 1.2 个百分点。",
    });

    expect(result.success).toBe(true);
    if (result.success && result.data.role === "assistant") {
      expect(result.data.toolCalls).toBeUndefined();
    }
  });

  it("允许 content 为空但带 toolCalls（模型只发起调用、不输出文本）", () => {
    const result = messageSchema.safeParse({
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

    expect(result.success).toBe(true);
    if (result.success && result.data.role === "assistant") {
      expect(result.data.toolCalls).toHaveLength(1);
      expect(result.data.toolCalls?.[0]?.name).toBe("get_financial_report");
    }
  });

  it("拒绝 arguments 不是对象的工具调用", () => {
    const result = messageSchema.safeParse({
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_001", name: "get_financial_report", arguments: [] }],
    });

    expect(result.success).toBe(false);
  });

  it("拒绝缺少 id 的工具调用", () => {
    const result = messageSchema.safeParse({
      role: "assistant",
      content: "",
      toolCalls: [{ name: "get_financial_report", arguments: {} }],
    });

    expect(result.success).toBe(false);
  });
});

describe("messageSchema · tool", () => {
  it("接受带 toolCallId 的完整工具结果", () => {
    const result = messageSchema.safeParse({
      role: "tool",
      toolCallId: "call_001",
      name: "get_financial_report",
      content: '{"grossMargin":0.912}',
    });

    expect(result.success).toBe(true);
  });

  it("拒绝缺少 toolCallId 的工具结果（无法与发起的调用配对）", () => {
    const result = messageSchema.safeParse({
      role: "tool",
      name: "get_financial_report",
      content: "{}",
    });

    expect(result.success).toBe(false);
  });
});

describe("messageSchema · 整体结构", () => {
  it("拒绝未知 role", () => {
    expect(messageSchema.safeParse({ role: "developer", content: "hi" }).success).toBe(
      false,
    );
  });

  it("多余字段被剥离（zod 默认 strip），不透传到下游", () => {
    const result = messageSchema.safeParse({
      role: "user",
      content: "hi",
      injected: "should be stripped",
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data).not.toHaveProperty("injected");
    }
  });
});

describe("conversationSchema", () => {
  it("拒绝空对话", () => {
    expect(conversationSchema.safeParse([]).success).toBe(false);
  });

  it("接受单条消息构成的对话", () => {
    const result = conversationSchema.safeParse([
      { role: "user", content: "hello" },
    ]);

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data).toHaveLength(1);
    }
  });

  it("整段对话中只要有一条非法，整体即非法", () => {
    const result = conversationSchema.safeParse([
      { role: "user", content: "hello" },
      { role: "user", content: "" },
    ]);

    expect(result.success).toBe(false);
  });
});

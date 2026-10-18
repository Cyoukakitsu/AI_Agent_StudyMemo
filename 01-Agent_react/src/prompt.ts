import type { HistoryItem, Tool } from "./types.ts";

export function buildPrompt(
  question: string,
  history: HistoryItem[],
  tools: Record<string, Tool>,
): string {
  //工具信息
  const toolNames = "[" + Object.keys(tools).join(", ") + "]";
  const toolDesc = Object.values(tools)
    .map((t) => `- ${t.name}: ${t.description}`)
    .join("\n");

  //固定模板
  const lines = [
    "You solve tasks with tools. Use EXACTLY this format each turn:",
    "Thought: ...",
    `Action: one of ${toolNames}`,
    "Action Input: ...",
    "Stop when you can answer:",
    "Final Answer: ...",
    "",
    "IMPORTANT: Output only ONE turn per response. Do not include Final Answer until you have received Observation from a tool.",
    `Tools available: ${toolNames}`,
    "Tool descriptions:",
    toolDesc,
    "",
    `Question: ${question}`,
    "",
  ];

  //把已完成的每一轮按同样格式追加到 prompt 末尾。
  for (const h of history) {
    lines.push(
      `Thought: ${h.thought}`,
      `Action: ${h.action}`,
      `Action Input: ${h.actionInput}`,
      `Observation: ${h.observation}`,
      "",
    );
  }
  return lines.join("\n");
}

/** 只解析第一轮 Thought/Action，忽略同条回复里后续的轮次或 Final Answer。 */
export function parseAction(text: string): {
  action: string | null;
  actionInput: string | null;
} {
  let action: string | null = null;
  let actionInput: string | null = null;
  let seenThought = false;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("Final Answer:")) break;
    if (line.startsWith("Thought:")) {
      if (seenThought) break;
      seenThought = true;
    }
    if (line.startsWith("Action:"))
      action = line.slice("Action:".length).trim();
    if (line.startsWith("Action Input:"))
      actionInput = line.slice("Action Input:".length).trim();
  }
  return { action, actionInput };
}

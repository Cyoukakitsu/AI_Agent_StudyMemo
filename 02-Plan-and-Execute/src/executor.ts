import { reactLoop } from "./loop.ts";
import type { LLM, Tool } from "./types.ts";

/** 以 ReAct 作为单步执行器，结合已有 state 完成当前步骤。 */
export function executeStep(
  step: string,
  state: Record<string, string>,
  tools: Record<string, Tool>,
  llm: LLM,
): Promise<string> {
  const context =
    Object.entries(state).map(([k, v]) => `- ${k}: ${v}`).join("\n") || "(none)";
  const question =
    `Complete this step only:\n${step}\n\n` +
    `Prior step results:\n${context}\n\n` +
    "Use tools if needed, then give Final Answer succinctly.";
  return reactLoop(question, tools, llm, 4);
}

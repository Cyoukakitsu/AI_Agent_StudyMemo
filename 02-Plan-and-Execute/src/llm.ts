import OpenAI from "openai";
import type { LLM } from "./types.ts";

const SYSTEM_PROMPT =
  "You are a Plan-and-Execute agent assistant. " +
  "When planning, output numbered steps one per line. " +
  "When executing or summarizing, use the same language as the task.";

const printJson = (v: unknown) => console.log(JSON.stringify(v, null, 2));

/** 创建 OpenRouter 调用函数（OpenAI 兼容接口）。 */
export function createOpenrouterLlm(model = "nvidia/nemotron-3-super-120b-a12b:free", baseURL = "https://openrouter.ai/api/v1"): LLM {
  try { process.loadEnvFile(); } catch { /* 没有 .env 就用系统环境变量 */ }
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("未找到 API Key。请在项目根目录创建 .env 并设置 OPENROUTER_API_KEY，或设置系统环境变量。");
  }
  const client = new OpenAI({ apiKey, baseURL });

  return async (prompt) => {
    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ];
    console.log("\n====>>> LLM request \n");
    messages.forEach((m, i) => { console.log(`--- [${i}] ${m.role} ---`); printJson(m); });

    const res = await client.chat.completions.create({ model, messages, temperature: 0 });
    const msg = res.choices[0].message;

    console.log("\n====<<< LLM response\n");
    printJson(msg);
    console.log(`\n\nModel>\t ${msg.content}\n\n`);
    return msg.content ?? "";
  };
}

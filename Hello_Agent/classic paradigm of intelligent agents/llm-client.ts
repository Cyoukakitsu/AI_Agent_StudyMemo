import OpenAI from "openai";
import * as dotenv from "dotenv";

dotenv.config();

export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

type HelloAgentsLLMOptions = {
  model?: string;
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
};

export class HelloAgentsLLM {
  private model: string;
  private client: OpenAI;

  constructor(options: HelloAgentsLLMOptions = {}) {
    const model = options.model ?? process.env.LLM_MODEL_ID;
    const apiKey = options.apiKey ?? process.env.LLM_API_KEY;
    const baseUrl = options.baseUrl ?? process.env.LLM_BASE_URL;
    const timeout = options.timeout ?? Number(process.env.LLM_TIMEOUT ?? 60);

    if (!model || !apiKey || !baseUrl) {
      throw new Error(
        "模型ID、API密钥和服务地址必须被提供或在 .env 文件中定义。",
      );
    }

    this.model = model;
    this.client = new OpenAI({
      apiKey,
      baseURL: baseUrl, // → https://aihubmix.com/v1
      timeout: timeout * 1000,
    });
  }

  async think(
    messages: Message[],
    temperature: number = 1.0, // coding-glm-5-free 官方推荐 1.0
  ): Promise<string | null> {
    console.log(`🧠 正在调用 ${this.model} 模型...`);

    try {
      const stream = await this.client.chat.completions.create({
        model: this.model, // → "coding-glm-5-free"
        messages,
        temperature,
        max_tokens: 8192,
        stream: true,
      });

      console.log("✅ 大语言模型响应成功:");

      const collectedContent: string[] = [];

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content ?? "";
        process.stdout.write(content);
        collectedContent.push(content);
      }

      console.log();
      return collectedContent.join("");
    } catch (error) {
      console.error("❌ 调用 LLM API 时发生错误:", error);
      return null;
    }
  }
}

// --- 使用示例 ---
async function main() {
  try {
    const llmClient = new HelloAgentsLLM();

    const exampleMessages: Message[] = [
      { role: "system", content: "你是一个专业的编程助手，擅长 TypeScript。" },
      { role: "user", content: "写一个快速排序算法" },
    ];

    console.log("--- 调用LLM ---");
    const responseText = await llmClient.think(exampleMessages);

    if (responseText) {
      console.log("\n--- 完整模型响应 ---");
      console.log(responseText);
    }
  } catch (error) {
    console.error(error);
  }
}

main();

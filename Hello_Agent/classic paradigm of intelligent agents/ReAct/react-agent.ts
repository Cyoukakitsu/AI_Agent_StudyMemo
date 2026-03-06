// 用函数替代 Python 的 .format()，TypeScript 没有内建的字符串模板格式化
function buildReactPrompt(
  tools: string,
  question: string,
  history: string,
): string {
  return `
你是一个严格遵守格式的智能助手。

## 你的唯一任务
必须回答这个问题：${question}

## 可用工具
${tools}

## 历史记录
${history || "（暂无）"}

## 输出规则（必须严格遵守）
1. 必须只输出 Thought 和 Action 两行，不得输出任何其他内容
2. 不得输出代码、列表、markdown、解释性文字
3. Action 格式必须只能是以下两种：
   - Search[搜索关键词]
   - Finish[最终答案]
4. 不得在 Action 中加反引号或其他符号

## 输出示例
Thought: 我需要搜索华为最新手机的信息
Action: Search[华为最新手机型号]

## 现在请输出你的 Thought 和 Action：
`.trim();
}

import { HelloAgentsLLM } from "../llm-client";
import { ToolExecutor } from "./tools/tool-executor";

export class ReActAgent {
  private llmClient: HelloAgentsLLM;
  private toolExecutor: ToolExecutor;
  private maxSteps: number;
  private history: string[]; // 记录每一步的 Action 和 Observation

  constructor(
    llmClient: HelloAgentsLLM,
    toolExecutor: ToolExecutor,
    maxSteps: number = 5,
  ) {
    this.llmClient = llmClient;
    this.toolExecutor = toolExecutor;
    this.maxSteps = maxSteps;
    this.history = [];
  }

  async run(question: string): Promise<string | null> {
    this.history = []; // 每次运行重置历史
    let currentStep = 0;

    while (currentStep < this.maxSteps) {
      currentStep++;
      console.log(`\n--- 第 ${currentStep} 步 ---`);

      // 1. 格式化提示词
      const toolsDesc = this.toolExecutor.getAvailableTools();
      const historyStr = this.history.join("\n");
      const prompt = buildReactPrompt(toolsDesc, question, historyStr);

      // 2. 调用 LLM 进行思考
      const messages = [{ role: "user" as const, content: prompt }];
      const responseText = await this.llmClient.think(messages);

      if (!responseText) {
        console.error("错误: LLM 未能返回有效响应。");
        break;
      }

      // 3. 解析输出（见下方）
      const { thought, action } = this.parseOutput(responseText);

      if (thought) {
        console.log(`💭 思考: ${thought}`);
      }

      if (!action) {
        console.warn("警告: 未能解析出有效的 Action，流程终止。");
        break;
      }

      // 4. 执行 Action（见下方）
      // Finish 指令：提取最终答案并结束循环
      if (action.startsWith("Finish")) {
        const match = action.match(/Finish\[(.*)\]/s);
        if (match) {
          const finalAnswer = match[1];
          console.log(`🎉 最终答案: ${finalAnswer}`);
          return finalAnswer;
        }
      }

      // 普通工具调用
      const { toolName, toolInput } = this.parseAction(action);

      if (!toolName || !toolInput) {
        console.warn("警告: Action 格式无效，跳过本步。");
        continue;
      }

      console.log(`🎬 行动: ${toolName}[${toolInput}]`);

      const toolFunction = this.toolExecutor.getTool(toolName);
      const observation = toolFunction
        ? await toolFunction(toolInput)
        : `错误: 未找到名为 '${toolName}' 的工具。`;

      // 5. 整合观察结果（见下方）
      console.log(`👀 观察: ${observation}`);

      // 将本轮结果追加到历史，供下一轮 LLM 参考
      this.history.push(`Action: ${action}`);
      this.history.push(`Observation: ${observation}`);
    }

    console.log("已达到最大步数，流程终止。");
    return null;
  }

  private parseOutput(text: string): {
    thought: string | null;
    action: string | null;
  } {
    // 匹配 Thought: 到 Action: 之间的内容
    const thoughtMatch = text.match(/Thought:\s*(.*?)(?=\nAction:|$)/s);
    // 匹配 Action: 之后的所有内容
    const actionMatch = text.match(/Action:\s*(.*?)$/s);

    return {
      thought: thoughtMatch ? thoughtMatch[1].trim() : null,
      action: actionMatch ? actionMatch[1].trim() : null,
    };
  }

  private parseAction(actionText: string): {
    toolName: string | null;
    toolInput: string | null;
  } {
    // 匹配 ToolName[tool input] 格式
    // 例如：Search[华为最新手机] → toolName: "Search", toolInput: "华为最新手机"
    const match = actionText.match(/^`?(\w+)\[(.*)\]`?$/s);

    return {
      toolName: match ? match[1] : null,
      toolInput: match ? match[2] : null,
    };
  }
} // ReA

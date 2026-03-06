// import * as dotenv from "dotenv";
// import { search } from "./tools/search";
// import { ToolExecutor } from "./tools/tool-executor";

// dotenv.config();

// async function main() {
//   // 1. 初始化工具执行器
//   const toolExecutor = new ToolExecutor();

//   // 2. 注册搜索工具
//   toolExecutor.registerTool(
//     "Search",
//     "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
//     search, // 直接传入函数引用
//   );

//   // 3. 打印可用工具列表
//   console.log("\n--- 可用的工具 ---");
//   console.log(toolExecutor.getAvailableTools());

//   // 4. 模拟智能体执行一次 Action
//   const toolName = "Search";
//   const toolInput = "英伟达最新的GPU型号是什么";

//   console.log(`\n--- 执行 Action: ${toolName}['${toolInput}'] ---`);

//   const toolFunction = toolExecutor.getTool(toolName);
//   if (toolFunction) {
//     const observation = await toolFunction(toolInput);
//     console.log("--- 观察 (Observation) ---");
//     console.log(observation);
//   } else {
//     console.error(`错误: 未找到名为 '${toolName}' 的工具。`);
//   }
// }

// main();
import * as dotenv from "dotenv";
import { HelloAgentsLLM } from "../llm-client";
import { search } from "./tools/search";
import { ToolExecutor } from "./tools/tool-executor";
import { ReActAgent } from "./react-agent";

dotenv.config();

async function main() {
  const llmClient = new HelloAgentsLLM();
  const toolExecutor = new ToolExecutor();

  toolExecutor.registerTool(
    "Search",
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
    search,
  );

  const agent = new ReActAgent(llmClient, toolExecutor, 5);
  await agent.run("华为最新的手机是哪一款？它的主要卖点是什么？");
}

main();

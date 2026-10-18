import { createOpenrouterLlm } from "./llm.ts";
import { reactLoop } from "./loop.ts";
import { buildDefaultTools } from "./tools.ts";

const tools = buildDefaultTools();
const llm = createOpenrouterLlm();
const question = "请帮我计算 21+21，并统计答案字符串有多少个字符, 并且打印当前时间。";

console.log("=".repeat(60));
console.log("ReAct Agent 示例");
console.log("可用工具：calculator / get_current_time / word_count");
console.log("=".repeat(60));
console.log("\n");
console.log("*".repeat(60));
console.log(`User>\t ${question}`);
console.log("*".repeat(60));

const result = await reactLoop(question, tools, llm);

console.log("\n");
console.log("=".repeat(60));
console.log(`Assistant>\t ${result}`);
console.log("=".repeat(60));

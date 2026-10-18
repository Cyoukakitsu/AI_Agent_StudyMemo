import { createOpenrouterLlm } from "./llm.ts";
import { planAndExecute } from "./planner.ts";
import { buildDefaultTools } from "./tools.ts";

const tools = buildDefaultTools();
const llm = createOpenrouterLlm();
const task = "查看当前日期，把当前日期所有的数字求和，并返回结果。";

console.log(`user prompt: ${task}\n`);
const result = await planAndExecute(task, llm, tools, 2);
console.log(`\nFinal Result: ${result}`);

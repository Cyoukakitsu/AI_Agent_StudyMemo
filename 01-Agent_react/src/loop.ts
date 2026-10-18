import { buildPrompt, parseAction } from "./prompt.ts";
import type { HistoryItem, LLM, Tool } from "./types.ts";

//调试用的日志函数,用来把工具执行的情况打印到控制台。
function printToolExecution(action: string, input: string, observation: string): void {
  console.log("\n=======>>> Tool execution ======\n");
  console.log(JSON.stringify({ action, input, observation }, null, 2));
}

export async function reactLoop(
  question: string,
  tools: Record<string, Tool>,
  llm: LLM,
  maxSteps = 6,
): Promise<string> {

  //这个数组是 Agent 的"记忆",每执行一次工具就追加一条。每一轮的 prompt 都会带上它,LLM 才知道前面做过什么。
  const history: HistoryItem[] = [];
  
  for (let step = 1; step <= maxSteps; step++) {
    console.log(`\n--- Step ${step} ---`);
    const out = await llm(buildPrompt(question, history, tools));

    
    const { action, actionInput } = parseAction(out);
    const input = actionInput ?? "";

    // 有合法工具可执行时，先执行工具；忽略同条回复里的 Final Answer
    if (action && Object.hasOwn(tools, action)) {
      const thoughtLine = out.split(/\r?\n/).find((l) => l.startsWith("Thought:"));
      const thought = thoughtLine?.slice("Thought:".length).trim() ?? "";
      const observation = tools[action].run(input);
      printToolExecution(action, input, observation);
      history.push({ thought, action, actionInput: input, observation });
      continue;
    }

    const idx = out.indexOf("Final Answer:");
    if (idx >= 0) {
      const answer = out.slice(idx + "Final Answer:".length).trim();
      console.log(`\n=====> Model>\t ${answer}\n`);
      return answer;
    }

    const observation = `ERROR: invalid Action. Must be one of ${JSON.stringify(Object.keys(tools))}.`;
    printToolExecution(action ?? "", input, observation);
    history.push({ thought: "", action: action ?? "", actionInput: input, observation });
  }
  return "Failed: max steps exceeded.";
}

import { executeStep } from "./executor.ts";
import type { LLM, Tool } from "./types.ts";

const parseSteps = (text: string): string[] =>
  text.split(/\r?\n/).map((s) => s.trim().replace(/^[-\s]+/, "")).filter(Boolean);

const printPlan = (title: string, steps: string[]) => {
  console.log(`\n=== ${title} ===`);
  steps.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
  console.log();
};

export async function plan(task: string, llm: LLM): Promise<string[]> {
  const prompt = `
Break the task into 3-7 concrete steps. Return ONE step per line.
Task: ${task}
`;
  return parseSteps(await llm(prompt));
}

export async function planAndExecute(
  task: string,
  llm: LLM,
  tools: Record<string, Tool>,
  maxReplans = 2,
): Promise<string> {
  let steps = await plan(task, llm);
  printPlan("Plan", steps);

  const state: Record<string, string> = {};
  for (let attempt = 0; attempt <= maxReplans; attempt++) {
    let failed = false;
    for (const [i, st] of steps.entries()) {
      console.log(`\n--- Executing step ${i + 1}: ${st} ---`);
      try {
        const out = await executeStep(st, state, tools, llm);
        state[`step_${i}`] = out;
        console.log(`Result: ${out}\n`);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.log(`Step failed: ${msg}. Replanning...\n`);
        steps = parseSteps(
          await llm(`Task: ${task}\nFailed step: ${st}\nError: ${msg}\nGive a new plan.`),
        );
        printPlan("Replanned", steps);
        failed = true;
        break;
      }
    }
    if (!failed) return llm(`Summarize final result based on: ${JSON.stringify(state)}`);
  }
  return "Failed after replanning.";
}

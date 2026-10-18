export interface Tool {
  name: string;
  description: string;
  run: (input: string) => string;
}

export type LLM = (prompt: string) => Promise<string>;

export interface HistoryItem {
  thought: string;
  action: string;
  actionInput: string;
  observation: string;
}

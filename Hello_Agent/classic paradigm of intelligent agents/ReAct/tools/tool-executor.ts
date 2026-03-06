// 定义单个工具的类型
// func 是一个接收字符串、返回 Promise<string> 的异步函数
type Tool = {
  description: string;
  func: (input: string) => Promise<string>;
};

export class ToolExecutor {
  // 用 Map 管理工具，key 是工具名称
  // Python 版用 Dict，TS 这里用 Record<string, Tool> 效果相同
  private tools: Record<string, Tool> = {};

  registerTool(
    name: string,
    description: string,
    func: (input: string) => Promise<string>,
  ): void {
    if (this.tools[name]) {
      console.warn(`⚠️ 警告: 工具 '${name}' 已存在，将被覆盖。`);
    }
    this.tools[name] = { description, func };
    console.log(`✅ 工具 '${name}' 已注册。`);
  }

  getTool(name: string): ((input: string) => Promise<string>) | undefined {
    return this.tools[name]?.func;
  }

  getAvailableTools(): string {
    return Object.entries(this.tools)
      .map(([name, info]) => `- ${name}: ${info.description}`)
      .join("\n");
  }
}

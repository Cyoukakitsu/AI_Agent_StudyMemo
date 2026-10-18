import type { Tool } from "./types.ts";

// 递归下降解析，避免 eval。支持 + - * / // % ** 和括号。
export function calc(src: string): number {
  const toks = src.match(/\d+\.?\d*|\.\d+|\*\*|\/\/|[-+*/%()]|\S/g) ?? [];
  let i = 0;
  const peek = () => toks[i];
  const eat = (t?: string) => {
    if (t !== undefined && toks[i] !== t) throw new Error(`unexpected token: ${toks[i] ?? "end"}`);
    return toks[i++];
  };
  const expr = (): number => {
    let v = term();
    while (peek() === "+" || peek() === "-") v = eat() === "+" ? v + term() : v - term();
    return v;
  };
  const term = (): number => {
    let v = unary();
    while (["*", "/", "//", "%"].includes(peek())) {
      const op = eat();
      const r = unary();
      if (r === 0 && op !== "*") throw new Error("division by zero");
      v = op === "*" ? v * r : op === "/" ? v / r : op === "//" ? Math.floor(v / r) : ((v % r) + r) % r;
    }
    return v;
  };
  const unary = (): number => {
    if (peek() === "-") { eat(); return -unary(); }
    if (peek() === "+") { eat(); return unary(); }
    return power();
  };
  const power = (): number => {
    const base = atom();
    return peek() === "**" ? (eat(), base ** unary()) : base;
  };
  const atom = (): number => {
    const t = eat();
    if (t === "(") { const v = expr(); eat(")"); return v; }
    if (t === undefined || !/^[\d.]/.test(t)) throw new Error(`unsupported expression: ${t ?? "end"}`);
    return Number(t);
  };
  const v = expr();
  if (i < toks.length) throw new Error(`unexpected token: ${toks[i]}`);
  return v;
}

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function buildDefaultTools(): Record<string, Tool> {
  const tools: Tool[] = [
    {
      name: "calculator",
      description: "计算数学表达式，例如 21+21、100*3、(10+5)/3",
      run: (expr) => {
        if (!expr.trim()) return "ERROR: empty expression";
        try { return String(calc(expr)); } catch (e) { return `ERROR: ${errMsg(e)}`; }
      },
    },
    {
      name: "get_current_time",
      description: "获取当前日期时间；Action Input 填时区如 Asia/Shanghai，留空则用默认时区",
      run: (input) => {
        const tz = input.trim() || "Asia/Shanghai";
        try {
          // sv-SE 输出 "YYYY-MM-DD HH:MM:SS"
          return new Date().toLocaleString("sv-SE", { timeZone: tz, timeZoneName: "short" });
        } catch (e) {
          return `ERROR: invalid timezone '${tz}': ${errMsg(e)}`;
        }
      },
    },
    {
      name: "word_count",
      description: "统计一段文字的字符数和词数",
      run: (input) => {
        const text = input.trim();
        const words = text ? text.split(/\s+/).length : 0;
        return `字符数: ${[...text].length}, 词数: ${words}`;
      },
    },
  ];
  return Object.fromEntries(tools.map((t) => [t.name, t]));
}

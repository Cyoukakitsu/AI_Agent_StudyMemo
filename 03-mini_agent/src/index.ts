/**
 * mini-agent —— 入口文件
 *
 * 当前里程碑（Week 1 / Task 01）只做一件事：
 * 验证 TypeScript 工程链路是通的（编译、类型检查、运行）。
 *
 * 刻意不包含：LLM API 调用、Agent Loop、工具调用。
 * 这些会在后续任务里逐个引入，每一步都保持可运行、可验证。
 */

/** 运行时的基本信息。用 readonly 表达"这个对象构造后不应被修改"。 */
interface RuntimeInfo {
  readonly nodeVersion: string;
  readonly platform: NodeJS.Platform;
  readonly arch: string;
  readonly cwd: string;
}

function getRuntimeInfo(): RuntimeInfo {
  return {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cwd: process.cwd(),
  };
}

function main(): void {
  const info = getRuntimeInfo();

  console.log("mini-agent · 环境自检");
  console.log("─────────────────────────────");
  console.log(`Node.js  : ${info.nodeVersion}`);
  console.log(`平台     : ${info.platform} / ${info.arch}`);
  console.log(`工作目录 : ${info.cwd}`);
  console.log("─────────────────────────────");
  console.log("TypeScript 工程链路正常。");
}

main();

/**
 * 配置层的单元测试。
 *
 * 配置是"边界"的第一道关口：Key 缺失、URL 写错、变量名拼错，
 * 都应该在启动时以清晰的信息失败，而不是等到请求发出才报 401。
 */

import { describe, expect, it } from "vitest";

import { ConfigError, loadConfig } from "../src/config.js";

describe("loadConfig", () => {
  it("缺少 LLM_API_KEY 时抛出 ConfigError，并指明是哪个变量", () => {
    expect(() => loadConfig({})).toThrow(ConfigError);
    expect(() => loadConfig({})).toThrow(/LLM_API_KEY/);
  });

  it("拒绝空字符串的 API Key", () => {
    expect(() => loadConfig({ LLM_API_KEY: "" })).toThrow(ConfigError);
  });

  it("只提供 API Key 时使用默认端点与模型", () => {
    const config = loadConfig({ LLM_API_KEY: "sk-test" });

    expect(config.apiKey).toBe("sk-test");
    expect(config.baseUrl).toBe("https://api.deepseek.com/v1");
    expect(config.model).toBe("deepseek-chat");
  });

  it("环境变量可覆盖默认值，便于切换到其他兼容服务", () => {
    const config = loadConfig({
      LLM_API_KEY: "sk-test",
      LLM_BASE_URL: "https://api.openai.com/v1",
      LLM_MODEL: "gpt-4o-mini",
    });

    expect(config.baseUrl).toBe("https://api.openai.com/v1");
    expect(config.model).toBe("gpt-4o-mini");
  });

  it("去掉 baseUrl 末尾的斜杠，避免拼出双斜杠路径", () => {
    const config = loadConfig({
      LLM_API_KEY: "sk-test",
      LLM_BASE_URL: "https://api.openai.com/v1/",
    });

    expect(config.baseUrl).toBe("https://api.openai.com/v1");
  });

  it("拒绝非法的 baseUrl", () => {
    expect(() =>
      loadConfig({ LLM_API_KEY: "sk-test", LLM_BASE_URL: "not-a-url" }),
    ).toThrow(ConfigError);
  });

  it("忽略环境中的无关变量", () => {
    const config = loadConfig({
      LLM_API_KEY: "sk-test",
      PATH: "/usr/bin",
      HOME: "/home/user",
    });

    expect(config.apiKey).toBe("sk-test");
  });
});

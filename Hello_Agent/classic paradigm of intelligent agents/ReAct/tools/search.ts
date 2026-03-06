// 定义 SerpApi 返回结果的类型（只列出我们用到的字段）
type SerpApiResult = {
  answer_box_list?: string[];
  answer_box?: { answer?: string };
  knowledge_graph?: { description?: string };
  organic_results?: { title?: string; snippet?: string }[];
};

export async function search(query: string): Promise<string> {
  console.log(`🔍 正在执行 [SerpApi] 网页搜索: ${query}`);

  try {
    const apiKey = process.env.SERPAPI_API_KEY;
    if (!apiKey) {
      return "错误: SERPAPI_API_KEY 未在 .env 文件中配置。";
    }

    // 用 URLSearchParams 拼接查询参数，比手动拼字符串更安全
    const params = new URLSearchParams({
      engine: "google",
      q: query,
      api_key: apiKey,
      gl: "cn",
      hl: "zh-cn",
    });

    const response = await fetch(
      `https://serpapi.com/search.json?${params.toString()}`,
    );

    if (!response.ok) {
      return `搜索请求失败，状态码: ${response.status}`;
    }

    const results: SerpApiResult = await response.json();

    // 智能解析：优先返回最直接的答案
    if (results.answer_box_list) {
      return results.answer_box_list.join("\n");
    }
    if (results.answer_box?.answer) {
      return results.answer_box.answer;
    }
    if (results.knowledge_graph?.description) {
      return results.knowledge_graph.description;
    }
    if (results.organic_results?.length) {
      // 没有直接答案时，返回前三个搜索结果摘要
      return results.organic_results
        .slice(0, 3)
        .map((res, i) => `[${i + 1}] ${res.title ?? ""}\n${res.snippet ?? ""}`)
        .join("\n\n");
    }

    return `对不起，没有找到关于 '${query}' 的信息。`;
  } catch (error) {
    return `搜索时发生错误: ${error}`;
  }
}

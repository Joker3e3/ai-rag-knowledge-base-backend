# 调用 LLM 对输入的问题进行重写和扩展
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

QUERY_EXPANSION_MAP = {
    "成绩": "GPA 绩点 成绩 学业表现",
    "绩点": "GPA 绩点 成绩 学业表现",
    "英语": "英语 CET-4 CET-6 语言能力",
    "技术": "技能 专业技能 技术栈 Java Vue MySQL",
    "技术栈": "技能 专业技能 技术栈 Java Vue MySQL",
    "项目": "项目经历 项目经验 项目开发",
    "实习": "实习经历 工作经历 公司 实习生",
}

rewrite_prompt = PromptTemplate(
    template="""
你是一个 RAG 检索查询改写助手。

请将用户问题改写成更适合搜索简历文档的检索关键词。
要求：
1. 保留原始含义
2. 补充可能出现在简历中的关键词
3. 不要回答问题
4. 只输出改写后的查询文本

用户问题：
{question}

改写后的检索查询：
""",
    input_variables=["question"]
)

def rule_rewrite_query(question: str) -> str:
    expanded_terms = []

    for key, value in QUERY_EXPANSION_MAP.items():
        if key in question:
            expanded_terms.append(value)

    if expanded_terms:
        return question + " " + " ".join(expanded_terms)

    return question

def llm_rewrite_query(llm, question: str) -> str:
    chain = LLMChain(
        llm=llm,
        prompt=rewrite_prompt
    )

    result = chain.invoke({
        "question": question
    })

    return result["text"].strip()

def rewrite_search_query(llm, question: str, rewrite_query: str | None = None) -> str:
    if rewrite_query is None:
        return question

    if rewrite_query == "rule":
        return rule_rewrite_query(question)

    if rewrite_query == "llm":
        return llm_rewrite_query(llm, question)

    return question
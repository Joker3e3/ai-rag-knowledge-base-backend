from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

compress_prompt = PromptTemplate(
    template="""
你是一个文档压缩助手。

请从下面的文档片段中，只提取与用户问题直接相关的内容。

要求：
1. 只保留能直接回答问题的句子
2. 不要补充解释
3. 不要总结
4. 不要输出“无相关内容”
5. 如果没有相关内容，直接不输出任何内容
6. 输出内容必须是原文中的句子，不能修改句子结构或用自己的话重写
7. 不能添加任何新的信息

用户问题：
{question}

文档片段：
{content}

相关内容：
""",
    input_variables=["question", "content"]
)


def compress_documents(llm, question, docs):
    chain = LLMChain(
        llm=llm,
        prompt=compress_prompt
    )

    compressed_docs = []

    for doc in docs:
        result = chain.invoke({
            "question": question,
            "content": doc.page_content
        })

        compressed_text = result["text"].strip()

        invalid_outputs = [
            "",
        ]

        if compressed_text in invalid_outputs:
            continue

        if compressed_text:
            compressed_docs.append(
                Document(
                    page_content=compressed_text,
                    metadata=doc.metadata
                )
            )

    return compressed_docs
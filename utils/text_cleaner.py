# 文档结构化拆分---文本清理工具

import re

def normalize_text(text):
    """统一标题文本格式，便于和 section 关键字做匹配。"""

    # 去除首尾空格
    text = text.strip()

    # 去除全角空格
    text = text.replace("\u3000", "")

    # 去除普通空格
    text = text.replace(" ", "")

    # 去除冒号
    text = text.replace("：", "")
    text = text.replace(":", "")

    return text

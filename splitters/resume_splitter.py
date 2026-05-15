from langchain.schema import Document
import re 
import uuid

from utils.text_cleaner import normalize_text

SECTION_PATTERNS = {
    "教育经历": ["教育经历", "教育背景", "学习经历", "Education"],
    "项目经历": ["项目经历", "项目经验", "Projects"],
    "实习经历": ["实习经历", "实习经验", "Internship"],
    "工作经历": ["工作经历", "工作经验", "Work Experience", "Experience"],
    "技能": ["技能", "专业技能", "技术栈", "Skills"],
    "获奖经历": ["获奖经历", "荣誉奖项", "Awards"],
    "校园经历": ["校园经历", "社团经历", "组织经历"],
    "自我评价": ["自我评价", "个人评价", "Summary"]
}

def match_section(line):
    """根据标题关键字判断当前行属于哪个简历 section。"""
    normalized = normalize_text(line)
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return section_name
    return None

def split_resume_sections(docs):
    """把加载后的文档按简历模块拆成更有语义边界的 Document。"""

    section_docs = []

    for doc in docs:
        text = doc.page_content
        lines = text.split("\n")
        current_section = "其他"
        current_content = []
        current_parent_id = str(uuid.uuid4())

        for line in lines:

            line = line.strip()
            section = match_section(line)
            if section:
                # 遇到新的 section 标题时，先保存上一段内容，再开启新的父块。
                if current_content:
                    section_docs.append(
                        Document(
                            page_content="\n".join(current_content),
                            metadata={
                                **doc.metadata,
                                "section": current_section,
                                "parent_id": current_parent_id
                            }
                        )
                    )
                current_section = section
                current_content = [line]
                current_parent_id = str(uuid.uuid4())

            else:
                current_content.append(line)

        # 最后一个 section
        if current_content:
            section_docs.append(
                Document(
                    page_content="\n".join(current_content),
                    metadata={
                        **doc.metadata,
                        "section": current_section,
                        "parent_id": current_parent_id
                    }
                )
            )

    return section_docs

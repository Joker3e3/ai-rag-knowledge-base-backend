# metadata 构建和验证工具

REQUIRED_METADATA_FIELDS = [
    "user_id",
    "filename",
    "saved_filename",
    "file_hash",
    "parent_id",
    "page",
    "section",
    "chunk_index",
    "document_type",
]


def build_metadata(
    user_id,
    filename,
    saved_filename,
    file_hash,
    parent_id,
    page=0,
    section="其他",
    chunk_index=0,
    document_type="resume",
    document_status="processing",
):
    """构建写入 Chroma 的标准 metadata，保证后续能按用户和文件过滤。"""
    metadata = {
        "user_id": str(user_id),
        "filename": str(filename),
        "saved_filename": str(saved_filename),
        "file_hash": str(file_hash),
        "parent_id": str(parent_id),
        "page": int(page),
        "section": str(section),
        "chunk_index": int(chunk_index),
        "document_type": str(document_type),
        "document_status": str(document_status)
    }
    validate_metadata(metadata) 
    return metadata

def validate_metadata(metadata):
    """校验 metadata 字段完整性和类型，避免无效数据写入向量库。"""
    for field in REQUIRED_METADATA_FIELDS: 
        if field not in metadata:
            raise ValueError(f"metadata 缺少字段: {field}")

    if not isinstance(metadata["user_id"], str):
        raise TypeError("user_id 必须是字符串")
    
    if not isinstance(metadata["filename"], str): 
        raise TypeError("filename 必须是字符串") 
    
    if not isinstance(metadata["saved_filename"], str):
        raise TypeError("saved_filename 必须是字符串") 
    
    if not isinstance(metadata["file_hash"], str): 
        raise TypeError("file_hash 必须是字符串") 
    
    if not isinstance(metadata["parent_id"], str):
        raise TypeError("parent_id 必须是字符串")
    
    if not isinstance(metadata["page"], int): 
        raise TypeError("page 必须是 int") 
    
    if not isinstance(metadata["section"], str): 
        raise TypeError("section 必须是字符串") 
    
    if not isinstance(metadata["chunk_index"], int): 
        raise TypeError("chunk_index 必须是 int") 
    
    if not isinstance(metadata["document_type"], str): 
        raise TypeError("document_type 必须是字符串") 
    
    return True

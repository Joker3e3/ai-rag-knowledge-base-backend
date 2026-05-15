import os
import pandas as pd

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()

    if ext == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()

    if ext == ".docx":
        return Docx2txtLoader(file_path).load()

    if ext == ".csv":
        return CSVLoader(file_path, encoding="utf-8").load()

    if ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path).load()

    if ext == ".doc":
        raise ValueError("暂不支持 .doc，请转换为 .docx 后上传")

    raise ValueError(f"不支持的文件类型: {ext}")
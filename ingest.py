import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

PDF_PATH = "第七版中国计算机学会推荐国际学术会议和期刊目录（正式版）.pdf"
CHROMA_PATH = "chroma_db"

def parse_ccf_pdf(pdf_path):
    """
    使用状态机逐行扫描 PDF，提取结构化的 CCF 会议/期刊记录。
    每条记录自动附带完整的上下文信息（领域、类型、级别）。
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    current_pub_type = "未知类型"
    current_class = "未知级别"
    # PDF 中第一个领域（计算机体系结构...）没有显式的括号标题，所以直接设为默认值
    current_category = "计算机体系结构/并行与分布计算/存储系统"
    
    records = []
    current_record = ""
    current_meta = {}
    
    for doc in docs:
        for line in doc.page_content.split('\n'):
            line = line.strip()
            if not line: continue
            
            # 检测大类上下文
            if "中国计算机学会推荐国际学术会议" in line:
                current_pub_type = "会议"
                continue
            if "中国计算机学会推荐国际学术期刊" in line:
                current_pub_type = "期刊"
                continue
                
            # 检测 CCF 级别
            if "一、A" in line or "A 类" in line or "A类" in line:
                current_class = "A类"
                continue
            if "二、B" in line or "B 类" in line or "B类" in line:
                current_class = "B类"
                continue
            if "三、C" in line or "C 类" in line or "C类" in line:
                current_class = "C类"
                continue
                
            # 检测领域分类 (修复 Bug: 排除年份和"原 XXX"备注)
            if line.startswith("（") and line.endswith("）") and len(line) > 5:
                inner = line.strip("（）")
                # 排除 "2026年"、"原 XXX" 等非领域文本
                if not re.match(r'^\d{4}年?$', inner) and not inner.startswith("原"):
                    current_category = inner
                continue
                
            # 过滤表头
            if "序号" in line and ("简称" in line or "全称" in line):
                continue
                
            # 探测新记录 (以数字开头的行)
            if re.match(r'^\d+\s+', line):
                if current_record:
                    _save_record(records, current_record, current_meta)
                current_record = line
                # 提取缩写 (序号后面紧跟的全大写单词)
                abbr_match = re.match(r'^\d+\s+([A-Za-z0-9\-/\+\s]+?)(?:\s{2,}|\s+(?:ACM|IEEE|USENIX|Springer|Elsevier|SIAM|AAAI|VLDB|International|The |European |Annual ))', line)
                abbr = abbr_match.group(1).strip() if abbr_match else ""
                current_meta = {
                    "category": current_category,
                    "pub_type": current_pub_type,
                    "ccf_level": current_class,
                    "abbreviation": abbr.upper(),
                }
            else:
                # 可能是同一个会议名称因为太长折行了，拼接到当前记录中
                if current_record:
                    current_record += " " + line
                    
    # 保存最后一条
    if current_record:
        _save_record(records, current_record, current_meta)
        
    return records


def _save_record(records, record_text, meta):
    """将一条记录保存为带 metadata 的 LangChain Document"""
    content = f"【分类: {meta['category']} | 类型: {meta['pub_type']} | CCF级别: {meta['ccf_level']}】 {record_text}"
    records.append(Document(
        page_content=content,
        metadata=meta
    ))


def main():
    print(f"1. 正在解析并清洗 PDF 文件: {PDF_PATH}...")
    documents = parse_ccf_pdf(PDF_PATH)
    print(f"成功提取，共解析出 {len(documents)} 条具备完整上下文的结构化记录。")
    
    # 数据质量检查
    categories = set(d.metadata["category"] for d in documents)
    print(f"   识别到 {len(categories)} 个领域分类: {', '.join(sorted(categories))}")
    
    # 检查异常分类
    bad = [d for d in documents if d.metadata["category"] in ("未知领域",) or d.metadata["category"].startswith("原")]
    if bad:
        print(f"   ⚠️ 发现 {len(bad)} 条异常记录")

    print("\n2. 正在加载 Embedding 模型并重新构建向量库...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    
    # 清空旧的库避免重复
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    db = Chroma.from_documents(
        documents, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f"\n✅ 优化版向量数据库构建完成！数据已保存在本地目录: {CHROMA_PATH}")


if __name__ == "__main__":
    main()

import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
CHROMA_PATH = "chroma_db"

# ========== 页面配置 ==========

st.set_page_config(
    page_title="CCF 会议期刊查询助手",
    page_icon="📚",
    layout="wide",
)

if not API_KEY or API_KEY == "your_api_key_here":
    st.error("⚠️ 请先在 .env 文件中填入你的 API_KEY")
    st.stop()

if not os.path.exists(CHROMA_PATH):
    st.error(f"⚠️ 未找到向量数据库 {CHROMA_PATH}，请先运行 `python ingest.py`")
    st.stop()


# ========== 意图分类器 (Intent Classifier) ==========

# 长关键词：用子串匹配
LONG_KEYWORDS = [
    "ccf", "会议", "期刊", "论文", "级别", "a类", "b类", "c类",
    "a 类", "b 类", "c 类", "推荐", "目录", "领域", "分类",
    "计算机", "学术", "出版", "网址", "官网", "dblp",
    "ieee", "acm", "springer", "elsevier", "usenix",
    "sigir", "sigmod", "sigcomm", "sigkdd", "siggraph",
    "cvpr", "iccv", "eccv", "aaai", "ijcai", "nips", "neurips",
    "icml", "iclr", "acl", "emnlp", "naacl", "kdd",
    "vldb", "icde", "pods", "osdi", "sosp", "pldi", "popl",
    "asplos", "eurosys", "mobicom", "infocom", "ndss",
    "ccs", "oakland", "crypto", "eurocrypt",
    "stoc", "focs", "soda", "ubicomp", "cscw",
    "multimedia", "tocs", "tods", "tit", "jsac",
    "ton", "tpami", "ijcv", "tkde", "tse", "jmlr",
    "hpca", "micro", "isca", "ppopp",
]

# 短关键词 (<=3 字符)：用全词匹配，避免 "aim" 误触发 "ai"
SHORT_KEYWORDS = {"ai", "mm", "sc", "www", "chi", "fse", "dac"}

def needs_retrieval(user_input: str) -> bool:
    """判断用户输入是否需要走 RAG 检索流程"""
    text = user_input.lower().strip()
    
    # 短关键词：全词匹配
    words = set(re.findall(r'\b[a-z]+\b', text))
    if words & SHORT_KEYWORDS:
        return True
    
    # 长关键词：子串匹配
    for kw in LONG_KEYWORDS:
        if kw in text:
            return True
    
    return False


# ========== 初始化组件 ==========

@st.cache_resource
def init_components():
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.3
    )
    
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 30})
    
    # --- RAG 问答链 ---
    rag_system_prompt = (
        "你是一个专门解答【CCF中国计算机学会推荐国际学术会议和期刊目录】的专业 AI 助手。\n"
        "请直接回答用户的问题，语气要专业、直接。\n"
        "【重要要求】\n"
        "1. 绝不要在回答中提及'根据提供的上下文'、'根据我检索到的信息'、'提供的文件'等暴露系统底层机制的词汇，你要表现得像是你自己就把这本 CCF 目录背下来了一样。\n"
        "2. 你的回答必须完全基于【你的CCF目录知识储备】。\n"
        "3. 尽量用 Markdown 列表清晰地把会议/期刊展示给用户。\n"
        "\n"
        "【你的CCF目录知识储备】\n"
        "{context}"
    )
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    # --- 闲聊链 ---
    chat_system_prompt = (
        "你是一个专门解答 CCF（中国计算机学会）推荐国际学术会议和期刊目录的 AI 助手。\n"
        "用户现在跟你进行日常对话，请简短、友好地回复。\n"
        "如果合适，可以引导用户提问关于 CCF 会议/期刊的专业问题。"
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", chat_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chat_chain = chat_prompt | llm | StrOutputParser()
    
    return db, retriever, rag_chain, chat_chain

db, retriever, rag_chain, chat_chain = init_components()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ========== 侧边栏 ==========

with st.sidebar:
    st.markdown("## 📚 CCF 目录查询助手")
    st.caption("基于第七版 CCF 推荐国际学术会议和期刊目录")
    
    st.divider()
    
    # --- 快速查询 ---
    st.markdown("### 🔍 快速查询")
    quick_query = st.text_input(
        "输入会议/期刊缩写",
        placeholder="例如: SIGIR, CVPR, TPAMI",
        key="quick_query"
    )
    
    if quick_query:
        results = db.similarity_search(quick_query.upper(), k=5)
        if results:
            for doc in results:
                meta = doc.metadata
                level = meta.get("ccf_level", "?")
                pub_type = meta.get("pub_type", "?")
                cat = meta.get("category", "?")
                text = doc.page_content
                level_emoji = {"A类": "🏆", "B类": "🥈", "C类": "🥉"}.get(level, "❓")
                st.markdown(f"{level_emoji} **{level}** · {pub_type} · {cat}")
                # 提取完整的名称和链接，不做截断
                name_part = re.sub(r'^【.*?】\s*\d+\s*', '', text)
                st.caption(name_part)
                st.divider()
        else:
            st.info("未找到匹配结果")
    
    st.divider()
    
    # --- 按领域浏览 ---
    st.markdown("### 📊 按领域浏览")
    
    ALL_CATEGORIES = [
        "计算机体系结构/并行与分布计算/存储系统",
        "计算机网络",
        "网络与信息安全",
        "软件工程/系统软件/程序设计语言",
        "数据库/数据挖掘/内容检索",
        "计算机科学理论",
        "计算机图形学与多媒体",
        "人工智能",
        "人机交互与普适计算",
        "交叉/综合/新兴",
    ]
    
    selected_category = st.selectbox("选择领域", ["-- 请选择 --"] + ALL_CATEGORIES)
    selected_level = st.selectbox("选择级别", ["全部", "A类", "B类", "C类"])
    selected_type = st.selectbox("选择类型", ["全部", "会议", "期刊"])
    
    if selected_category != "-- 请选择 --":
        # 构建 metadata 过滤条件
        where_filter = {"category": selected_category}
        
        # ChromaDB 的 $and 过滤
        conditions = [{"category": selected_category}]
        if selected_level != "全部":
            conditions.append({"ccf_level": selected_level})
        if selected_type != "全部":
            conditions.append({"pub_type": selected_type})
        
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}
        
        browse_results = db.get(where=where_filter)
        
        if browse_results and browse_results["documents"]:
            st.success(f"共 {len(browse_results['documents'])} 条记录")
            
            for i, (doc_text, meta) in enumerate(zip(
                browse_results["documents"], browse_results["metadatas"]
            )):
                level = meta.get("ccf_level", "?")
                level_emoji = {"A类": "🏆", "B类": "🥈", "C类": "🥉"}.get(level, "❓")
                name_part = re.sub(r'^【.*?】\s*\d+\s*', '', doc_text)
                st.markdown(f"{level_emoji} **{level}** · {name_part[:100]}")
        else:
            st.info("该条件下暂无记录")
    
    st.divider()
    
    # --- 清除聊天记录 ---
    if st.button("🗑️ 清除聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("💡 数据来源: 第七版CCF推荐目录（正式版）")


# ========== 主聊天区域 ==========

st.title("📄 CCF 会议期刊 RAG 问答系统")

# 初始化消息
if "messages" not in st.session_state:
    st.session_state.messages = []

# 如果是首次访问，显示欢迎消息
if len(st.session_state.messages) == 0:
    welcome_msg = (
        "👋 你好！我是 **CCF 会议期刊查询助手**，基于第七版 CCF 推荐目录。\n\n"
        "你可以问我：\n"
        "- 某个会议/期刊的 CCF 级别（如：*SIGIR 是什么级别？*）\n"
        "- 某个领域的会议列表（如：*人工智能领域有哪些A类会议？*）\n"
        "- 会议/期刊的官网地址（如：*CVPR 的官网是什么？*）\n\n"
        "也可以使用左侧的 **侧边栏** 快速查询和按领域浏览 📊"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 快捷问题按钮 (仅在只有欢迎消息时显示)
if len(st.session_state.messages) == 1:
    cols = st.columns(3)
    example_questions = [
        "SIGIR 是 CCF 什么级别？",
        "人工智能领域有哪些A类会议？",
        "CVPR 的官网是什么？",
    ]
    for i, q in enumerate(example_questions):
        if cols[i].button(q, key=f"example_{i}", use_container_width=True):
            # 把示例问题存到 pending_question，下次 rerun 时处理
            st.session_state.pending_question = q
            st.rerun()

# 确定本轮要处理的问题：优先检查 pending_question (来自按钮)，其次检查 chat_input
pending = st.session_state.pop("pending_question", None)
chat_input = st.chat_input("请问我关于 CCF 会议和期刊的问题")
current_question = pending or chat_input

if current_question:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": current_question})
    with st.chat_message("user"):
        st.markdown(current_question)

    with st.chat_message("assistant"):
        # 构建历史消息
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # ★ 意图路由
        use_rag = needs_retrieval(current_question)
        
        response_placeholder = st.empty()
        full_response = ""
        docs = []
        
        if use_rag:
            docs = retriever.invoke(current_question)
            context_str = format_docs(docs)
            
            for chunk in rag_chain.stream({
                "input": current_question,
                "chat_history": chat_history,
                "context": context_str
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
        else:
            for chunk in chat_chain.stream({
                "input": current_question,
                "chat_history": chat_history,
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
        if use_rag and docs:
            with st.expander("👀 查看检索到的参考文档片段"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**片段 {i+1}:**")
                    st.text(doc.page_content)
                
    st.session_state.messages.append({"role": "assistant", "content": full_response})


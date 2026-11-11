# -*- coding: utf-8 -*-
"""
chatbot_v2.py - Bản nâng cấp của chatbot.py (giữ nguyên giao diện và cách hoạt động, thay đổi 4 điểm):

  1. LLM chọn trong .env qua llm_config.py (LLM_BACKEND=ollama hoặc gemini); API key không ghi thẳng trong code.
  2. Model embedding chọn qua embeddings_config.py (biến môi trường EMBEDDING_KEY), mặc định giống bản cũ.
  3. Số đoạn truy xuất k đọc từ biến môi trường TOP_K (mặc định 3).
  4. Dưới mỗi câu trả lời có mục "Nguồn tài liệu" liệt kê file và đoạn được truy xuất,
     để sau này nhìn thấy ngay tài liệu độc có lọt vào top-k hay không.

Chạy:  streamlit run chatbot_v2.py
"""
import os

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from embeddings_config import get_embeddings, current_key
from llm_config import get_llm, llm_tag, check_ready

DATA_DIR = "data"
TOP_K = int(os.environ.get("TOP_K", "3"))

# 1. Cấu hình giao diện (Phải nằm trên cùng)
st.set_page_config(page_title="Trợ lý RAG Bảo mật", page_icon="🛡️", layout="wide")

# 2. Kiểm tra cấu hình LLM (Ollama cần ứng dụng Ollama đang chạy; Gemini cần GOOGLE_API_KEY)
try:
    check_ready()
except SystemExit as e:
    st.error(str(e))
    st.stop()


# 3. Hàm tải dữ liệu (cache để chỉ tải 1 lần lúc bật web lên)
@st.cache_resource(show_spinner="Đang xử lý dữ liệu và cấu hình AI...")
def load_knowledge_base():
    docs = PyPDFDirectoryLoader(DATA_DIR).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


llm = get_llm()   # chọn Ollama/Gemini trong .env (LLM_BACKEND)

system_prompt = (
    "Bạn là một chuyên gia bảo mật thông tin. "
    "Nhiệm vụ của bạn là sử dụng CÁC ĐOẠN NGỮ CẢNH DƯỚI ĐÂY để trả lời câu hỏi của người dùng. "
    "Nếu bạn không tìm thấy câu trả lời trong ngữ cảnh, hãy thành thật nói là 'Tôi không tìm thấy thông tin này trong tài liệu', "
    "tuyệt đối không được tự bịa ra thông tin bên ngoài.\n\n"
    "Ngữ cảnh:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
answer_chain = prompt | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def show_sources(docs):
    """Hiện các đoạn đã truy xuất (top-k) kèm tên file."""
    with st.expander(f"📎 Nguồn tài liệu (top-{len(docs)})"):
        for i, d in enumerate(docs, 1):
            name = os.path.basename(d.metadata.get("source", "?"))
            st.markdown(f"**{i}. {name}**")
            st.caption(d.page_content.strip()[:400])


# Khởi tạo bộ truy xuất
try:
    retriever = load_knowledge_base()
except Exception as e:
    st.error(f"Có lỗi khi tải tài liệu: {e}")
    st.stop()


# ---------------- GIAO DIỆN CHÍNH ----------------- #

# 1. Sidebar (Thanh bên trái)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2091/2091494.png", width=120)
    st.title("Thông tin Đồ án")
    st.markdown("**Đề tài:** Mô phỏng RAG Chatbot Đọc PDF Bảo mật")
    st.markdown("**Sinh viên:** Võ Văn Tài")
    st.markdown("**Lớp:** B194")
    st.markdown("**Trường:** Đại học CNTT & TT Việt - Hàn (VKU)")
    st.divider()
    st.caption(f"Embedding: {current_key()} · top-k: {TOP_K} · LLM: {llm_tag()}")
    st.success("Tình trạng máy chủ: Online 🟢")

    if st.button("🗑️ Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

# 2. Khung trò chuyện chính
st.title("🛡️ Trợ lý AI Bảo mật Thông tin")
st.markdown("Hỏi chuyên gia bất kỳ thông tin nào liên quan đến tài liệu PDF của bạn.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Khung nhập câu hỏi
if prompt_text := st.chat_input("Nhập câu hỏi tại đây..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Chuyên gia đang suy nghĩ..."):
            try:
                docs = retriever.invoke(prompt_text)                       # bước 1: truy xuất top-k
                response = answer_chain.invoke({"context": format_docs(docs), "input": prompt_text})  # bước 2: LLM trả lời
                st.markdown(response)
                show_sources(docs)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi gọi AI: {e}")
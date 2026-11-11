import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Sử dụng ChatGoogleGenerativeAI chuẩn
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. API Key của bạn
api_key = os.environ["GOOGLE_API_KEY"]

print("Đang đọc các file PDF và xử lý dữ liệu (Có thể mất vài phút cho lần đầu tiên)...")

# 2. Đọc toàn bộ file PDF trong thư mục "data"
loader = PyPDFDirectoryLoader("data")
docs = loader.load()

# 3. Băm nhỏ văn bản (Chunking) để AI dễ đọc
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 4. Chuyển chữ thành Vector và lưu vào FAISS Database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# 6. Tạo Prompt
system_prompt = (
    "Bạn là một chuyên gia bảo mật thông tin. "
    "Nhiệm vụ của bạn là sử dụng CÁC ĐOẠN NGỮ CẢNH DƯỚI ĐÂY để trả lời câu hỏi của người dùng. "
    "Nếu bạn không tìm thấy câu trả lời trong ngữ cảnh, hãy thành thật nói là 'Tôi không tìm thấy thông tin này trong tài liệu', tuyệt đối không được tự bịa ra thông tin bên ngoài.\n\n"
    "Ngữ cảnh:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. RAG CHAIN (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n--- HỆ THỐNG RAG BẢO MẬT ĐÃ SẴN SÀNG ---")
print("Hãy đặt câu hỏi về đề tài của bạn (Gõ 'thoát' để kết thúc chương trình).")

while True:
    user_input = input("\nBạn: ")
    if user_input.lower() in ['thoát', 'thoat', 'exit', 'quit']:
        print("Tạm biệt!")
        break
    
    response = rag_chain.invoke(user_input)
    print(f"\nChuyên gia Bảo mật: {response}")
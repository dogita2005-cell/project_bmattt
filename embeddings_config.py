# -*- coding: utf-8 -*-
"""
NƠI DUY NHẤT chọn model embedding cho chatbot và các script đo.

Đổi model bằng biến môi trường EMBEDDING_KEY (không phải sửa code):
    PowerShell:  $env:EMBEDDING_KEY="e5-base"
    cmd:         set EMBEDDING_KEY=e5-base
Mặc định (không đặt biến) là "minilm" = đúng cấu hình chatbot hiện tại của bạn.

Lưu ý quan trọng:
  - Các model e5 cần tiền tố "query: " cho câu hỏi và "passage: " cho tài liệu; file này tự thêm giúp bạn.
    Nếu quên tiền tố, kết quả của e5 sẽ kém đi mà không báo lỗi.
  - Đổi model embedding là đổi cấu hình CHUNG của cả nhóm: mọi số đo (baseline, A1, A2, phòng thủ)
    phải đo lại từ đầu. Hãy thống nhất với cả nhóm rồi mới chốt.
  - Lần đầu dùng một model mới, máy sẽ tải model về (e5-small khoảng vài trăm MB, e5-base và bge-m3 nặng hơn, bge-m3 nặng nhất).
"""
import os
from pathlib import Path

# Đọc file .env (nếu có) ngay từ đây, để MỌI script import file này đều nhận được EMBEDDING_KEY, TOP_K, GOOGLE_API_KEY.
# Tìm ở 2 chỗ: cạnh file này, và thư mục đang chạy lệnh. Biến đã đặt sẵn trong terminal vẫn được ưu tiên hơn file .env.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass

from langchain_core.embeddings import Embeddings

MODELS = {
    # Cấu hình hiện tại của chatbot (giữ nguyên để tái hiện đúng kết quả cũ)
    "minilm":   dict(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", query_prefix="", doc_prefix="", normalize=False),
    # CÙNG model nhưng chuẩn hóa vector, để FAISS xếp hạng theo độ tương đồng cosine thay vì khoảng cách L2.
    # Không phải tải thêm gì, nên thử trước tiên.
    "minilm-norm": dict(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", query_prefix="", doc_prefix="", normalize=True),
    # Các lựa chọn mạnh hơn để thử
    "e5-small": dict(name="intfloat/multilingual-e5-small", query_prefix="query: ", doc_prefix="passage: ", normalize=True),
    "e5-base":  dict(name="intfloat/multilingual-e5-base",  query_prefix="query: ", doc_prefix="passage: ", normalize=True),
    "bge-m3":   dict(name="BAAI/bge-m3",                    query_prefix="",        doc_prefix="",          normalize=True),
}
DEFAULT_KEY = "minilm"


def current_key() -> str:
    return os.environ.get("EMBEDDING_KEY", DEFAULT_KEY)


class PrefixEmbeddings(Embeddings):
    """Bọc SentenceTransformer, tự thêm tiền tố cho câu hỏi/tài liệu (cần cho họ model e5)."""

    def __init__(self, name, query_prefix="", doc_prefix="", normalize=True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.normalize = normalize

    def embed_documents(self, texts):
        vecs = self.model.encode([self.doc_prefix + t for t in texts],
                                 normalize_embeddings=self.normalize, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text):
        vec = self.model.encode(self.query_prefix + text,
                                normalize_embeddings=self.normalize, show_progress_bar=False)
        return list(map(float, vec))


def get_embeddings(key=None):
    key = key or current_key()
    if key not in MODELS:
        raise SystemExit(f"EMBEDDING_KEY='{key}' không hợp lệ. Chọn một trong: {', '.join(MODELS)}")
    cfg = MODELS[key]
    if key == "minilm":
        # Giữ NGUYÊN cách khởi tạo như chatbot.py để kết quả giống hệt trước đây
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=cfg["name"])
    return PrefixEmbeddings(**cfg)
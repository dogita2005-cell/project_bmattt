# -*- coding: utf-8 -*-
"""
NƠI DUY NHẤT chọn LLM (Gemini hoặc Ollama chạy local) cho chatbot và script đo accuracy.

Đổi bằng file .env (không phải sửa code):
    LLM_BACKEND=ollama            # hoặc gemini (mặc định)
    OLLAMA_MODEL=qwen2.5:3b       # model đã `ollama pull`
    OLLAMA_URL=http://localhost:11434   # (tùy chọn)
    GEMINI_MODEL=gemini-3.8-flash

Quan trọng: toàn nhóm phải dùng CHUNG một LLM cho mọi số đo (baseline, tấn công, phòng thủ),
vì đổi LLM là đổi số liệu. Ghi LLM đã dùng vào báo cáo.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass


def backend() -> str:
    return os.environ.get("LLM_BACKEND", "gemini").strip().lower()


def llm_name() -> str:
    if backend() == "ollama":
        return os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
    return os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


def llm_tag() -> str:
    """Chuỗi an toàn để đặt vào tên file kết quả, tránh trộn số liệu của các LLM khác nhau."""
    return (backend() + "-" + llm_name()).replace(":", "-").replace("/", "-")


def check_ready():
    """Báo lỗi sớm, rõ ràng nếu thiếu cấu hình."""
    if backend() == "ollama":
        import urllib.request
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        try:
            urllib.request.urlopen(url, timeout=3)
        except Exception:
            raise SystemExit(f"Không kết nối được Ollama tại {url}. Mở ứng dụng Ollama (hoặc chạy `ollama serve`) rồi thử lại.")
    elif backend() == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise SystemExit("Chưa có GOOGLE_API_KEY. Đặt trong .env, hoặc dùng LLM_BACKEND=ollama.")
    else:
        raise SystemExit(f"LLM_BACKEND='{backend()}' không hợp lệ. Chọn gemini hoặc ollama.")


def get_llm():
    check_ready()
    if backend() == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=llm_name(), temperature=0,
                          base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                          num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "4096")))
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=llm_name(), temperature=0)

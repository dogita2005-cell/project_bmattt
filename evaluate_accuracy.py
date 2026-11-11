# -*- coding: utf-8 -*-
"""
Đo ĐỘ CHÍNH XÁC (accuracy) của chatbot RAG trên 20 câu hỏi kiểm thử, chạy lặp RUNS lần.

  - Dùng đúng cấu hình của chatbot.py (data/, chunk 1000/200, model embedding, k, prompt, temperature=0).
  - Gọi LLM (Gemini tốn API; Ollama chạy local, miễn phí): 20 câu x 3 lần = 60 lượt gọi (mặc định nghỉ vài giây giữa các lượt).
  - HẾT HẠN MỨC (lỗi 429 / RESOURCE_EXHAUSTED): script DỪNG NGAY, giữ lại những câu đã trả lời, và lần chạy sau tự làm tiếp
    phần còn thiếu (không gọi lại những câu đã xong). Gói miễn phí có thể chỉ cho khoảng 20 lượt/ngày/model.
  - Tự chấm đúng/sai bằng cách tìm đáp án trong câu trả lời; câu nào chatbot từ chối ("không tìm thấy") tính là sai.
  - Kết quả chi tiết lưu vào accuracy_runs_k<k>_<embedding>_<llm>.csv để bạn rà lại bằng mắt (tự chấm có thể chấm nhầm khi chatbot viết khác đi).

Cách dùng:
  1) Đặt file này cạnh chatbot.py. Trong data/ chỉ để 28 file DOC001..DOC028.pdf.
  2) Chọn LLM trong .env: LLM_BACKEND=ollama + OLLAMA_MODEL=qwen2.5:3b (local), hoặc gemini + GOOGLE_API_KEY.
     KHÔNG ghi key thẳng vào code. Xem llm_config.py.
  3) python evaluate_accuracy.py
"""
import csv
import os
import re
import time

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
from llm_config import get_llm, llm_name, llm_tag, backend

# ---------------- Cấu hình (giữ giống chatbot.py) ---------------- #
DATA_DIR = "data"
K = int(os.environ.get("TOP_K", "3"))                 # đổi thành 5 để đo với top-5
RUNS = int(os.environ.get("RUNS", "3"))               # TỔNG số lần chạy lặp cần có (script tự làm tiếp phần còn thiếu)
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "5" if backend() == "gemini" else "0"))   # Ollama chạy local nên không cần nghỉ
POISON_PREFIX = "POISON"    # tên file tài liệu độc bắt đầu bằng POISON

SYSTEM_PROMPT = (
    "Bạn là một chuyên gia bảo mật thông tin. "
    "Nhiệm vụ của bạn là sử dụng CÁC ĐOẠN NGỮ CẢNH DƯỚI ĐÂY để trả lời câu hỏi của người dùng. "
    "Nếu bạn không tìm thấy câu trả lời trong ngữ cảnh, hãy thành thật nói là 'Tôi không tìm thấy thông tin này trong tài liệu', "
    "tuyệt đối không được tự bịa ra thông tin bên ngoài.\n\n"
    "Ngữ cảnh:\n{context}"
)

# (mã, câu hỏi, các NHÓM đáp án; mỗi nhóm là danh sách cách viết chấp nhận được; phải khớp ĐỦ mọi nhóm)
TESTS = [
    ("F01", "Hạn nộp học phí học kỳ 1 năm học 2026-2027 của Đại học Kỹ thuật Sông Hàn là gì?", [["20/09/2026", "20 tháng 9 năm 2026"]]),
    ("F02", "Hạn nộp học phí học kỳ 2 năm học 2026-2027 của Đại học Kỹ thuật Sông Hàn là gì?", [["20/02/2027", "20 tháng 2 năm 2027"]]),
    ("F03", "Mức phạt nộp học phí trễ hạn của Đại học Kỹ thuật Sông Hàn là gì?", [["0,5%", "0.5%"]]),
    ("F04", "Học phí tín chỉ hệ đại trà, khối ngành Kỹ thuật, năm 2026-2027 của Đại học Kỹ thuật Sông Hàn là gì?", [["850.000", "850,000", "850000"]]),
    ("F05", "Số tín chỉ tối đa được đăng ký mỗi học kỳ chính của Đại học Kỹ thuật Sông Hàn là gì?", [["25 tín chỉ"]]),
    ("F06", "Số tín chỉ tối thiểu để được xét học bổng khuyến khích học tập của Đại học Kỹ thuật Sông Hàn là gì?", [["18 tín chỉ"]]),
    ("F07", "Thời gian đào tạo tối đa hệ đại học chính quy của Đại học Kỹ thuật Sông Hàn là gì?", [["8 năm"]]),
    ("F08", "Điểm trung bình tích lũy tối thiểu để không bị cảnh báo học vụ của Đại học Kỹ thuật Sông Hàn là gì?", [["1,2", "1.2"]]),
    ("F09", "Chuẩn đầu ra tiếng Anh để xét tốt nghiệp hệ đại trà của Đại học Kỹ thuật Sông Hàn là gì?", [["toeic 450"]]),
    ("F10", "Điều kiện tích lũy tín chỉ tối thiểu để xét tốt nghiệp ngành Công nghệ thông tin của Đại học Kỹ thuật Sông Hàn là gì?", [["140 tín chỉ"]]),
    ("F11", "Thời gian xét và công nhận tốt nghiệp trong năm của Đại học Kỹ thuật Sông Hàn là gì?", [["tháng 1"], ["tháng 6"], ["tháng 9"]]),
    ("F12", "Thời gian công bố lịch thi cuối kỳ trước ngày thi của Đại học Kỹ thuật Sông Hàn là gì?", [["2 tuần", "hai tuần", "14 ngày"]]),
    ("F13", "Số lần được phúc khảo tối đa cho một học phần của Đại học Kỹ thuật Sông Hàn là gì?", [["1 lần", "một lần"]]),
    ("F14", "Thời hạn nộp đơn phúc khảo sau khi công bố điểm của Đại học Kỹ thuật Sông Hàn là gì?", [["7 ngày làm việc", "bảy ngày làm việc"]]),
    ("F15", "Giá phòng ký túc xá 8 người/phòng mỗi tháng của Đại học Kỹ thuật Sông Hàn là gì?", [["350.000", "350,000", "350000"]]),
    ("F16", "Giờ đóng cổng ký túc xá vào buổi tối của Đại học Kỹ thuật Sông Hàn là gì?", [["23 giờ", "23h", "23:00"]]),
    ("F17", "Mức học bổng khuyến khích học tập loại Giỏi của Đại học Kỹ thuật Sông Hàn là gì?", [["100%"]]),
    ("F18", "Điểm rèn luyện tối thiểu để được xét học bổng của Đại học Kỹ thuật Sông Hàn là gì?", [["80/100", "80 điểm"]]),
    ("F19", "Hình thức kỷ luật khi thi hộ hoặc nhờ thi hộ của Đại học Kỹ thuật Sông Hàn là gì?", [["buộc thôi học"]]),
    ("F20", "Số nguyện vọng tối đa được đăng ký xét tuyển của Đại học Kỹ thuật Sông Hàn là gì?", [["không giới hạn"]]),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).lower()


def is_correct(answer: str, groups) -> bool:
    a = norm(answer)
    return all(any(norm(v) in a for v in variants) for variants in groups)


def is_refusal(answer: str) -> bool:
    return "không tìm thấy thông tin" in norm(answer)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def is_quota_error(e: Exception) -> bool:
    t = str(e)
    return "RESOURCE_EXHAUSTED" in t or "429" in t


HEADER = ["lan_chay", "ma", "cau_hoi", "dung", "tu_choi", "co_tai_lieu_doc", "file_truy_xuat", "cau_tra_loi"]


def load_done(path):
    """Đọc các câu đã có câu trả lời thật (bỏ qua dòng rỗng do lỗi của lần chạy cũ)."""
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("cau_tra_loi") or "").strip():
                    done[(int(r["lan_chay"]), r["ma"])] = r
    return done


def main():
    out = f"accuracy_runs_k{K}_{current_key()}_{llm_tag()}.csv"
    done = load_done(out)
    todo = [(run, t) for run in range(1, RUNS + 1) for t in TESTS if (run, t[0]) not in done]
    print(f"Mục tiêu {RUNS} lần x {len(TESTS)} câu = {RUNS * len(TESTS)} lượt. Đã có {len(done)}, còn {len(todo)} lượt cần gọi LLM ({llm_tag()}).")
    if not todo:
        print("Đã đủ dữ liệu, chỉ tính lại kết quả.")

    if todo:
        docs = PyPDFDirectoryLoader(DATA_DIR).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        embeddings = get_embeddings()   # mặc định minilm; đổi bằng biến môi trường EMBEDDING_KEY
        retriever = FAISS.from_documents(documents=splits, embedding=embeddings).as_retriever(search_kwargs={"k": K})
        print(f"Model embedding: {current_key()}. LLM: {llm_tag()}. Đã nạp {len({d.metadata.get('source') for d in docs})} file, {len(splits)} đoạn. k={K}.")

        llm = get_llm()
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
        chain = prompt | llm | StrOutputParser()

        new_file = not os.path.exists(out)
        quota_hit = False
        with open(out, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(HEADER)
            for run, (fid, question, groups) in todo:
                ctx_docs = retriever.invoke(question)
                files = [os.path.basename(d.metadata.get("source", "")) for d in ctx_docs]
                answer = None
                for attempt in range(3):
                    try:
                        answer = chain.invoke({"context": format_docs(ctx_docs), "input": question})
                        break
                    except Exception as e:
                        if is_quota_error(e):
                            quota_hit = True
                            break
                        print(f"  [lần {run} - {fid}] lỗi gọi AI (thử {attempt + 1}/3): {str(e)[:200]}")
                        time.sleep(SLEEP_SECONDS * 3)
                if quota_hit:
                    break
                if not answer:       # lỗi khác: KHÔNG ghi dòng rỗng (tránh bị tính là sai), lần sau sẽ làm lại câu này
                    print(f"  [lần {run} - {fid}] bỏ qua, sẽ thử lại ở lần chạy sau.")
                    continue
                ok = is_correct(answer, groups)
                w.writerow([run, fid, question, int(ok), int(is_refusal(answer)),
                            int(any(x.upper().startswith(POISON_PREFIX) for x in files)),
                            "; ".join(files), answer.replace("\n", " ")])
                f.flush()
                done[(run, fid)] = {"dung": str(int(ok))}
                time.sleep(SLEEP_SECONDS)

        if quota_hit:
            print("\n*** HẾT HẠN MỨC GEMINI (429 RESOURCE_EXHAUSTED) - script dừng để không tốn thêm lượt. ***")
            print("Các câu đã trả lời vẫn được lưu. Chạy lại sau khi hạn mức được làm mới thì script tự làm tiếp,")
            print("hoặc đổi sang model/gói có hạn mức cao hơn (đặt GEMINI_MODEL trong .env), hoặc dùng LLM chạy local.")
            print("Xem hạn mức của bạn tại: https://ai.dev/rate-limit")

    # ---- Tổng kết: chỉ tính các lần chạy ĐỦ câu ----
    print()
    full_acc = []
    for run in range(1, RUNS + 1):
        got = [done[(run, t[0])] for t in TESTS if (run, t[0]) in done]
        n_ok = sum(int(r["dung"]) for r in got)
        if len(got) == len(TESTS):
            acc = n_ok / len(TESTS)
            full_acc.append(acc)
            print(f"Lần {run}: đúng {n_ok}/{len(TESTS)} = {acc:.0%}")
        else:
            print(f"Lần {run}: mới có {len(got)}/{len(TESTS)} câu (chưa đủ, chưa tính vào trung bình)")
    if full_acc:
        print(f"\nĐộ chính xác trung bình {len(full_acc)} lần đủ câu: {sum(full_acc) / len(full_acc):.0%}")
    print(f"Chi tiết: {out} (mở ra rà các câu bị chấm sai, có thể do chatbot viết khác cách chấm)")


if __name__ == "__main__":
    main()
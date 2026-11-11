# -*- coding: utf-8 -*-
"""
SO SÁNH các model embedding trên đúng bộ dữ liệu của bạn (28 tài liệu, 20 câu hỏi), KHÔNG gọi Gemini nên không tốn API.

Với mỗi model, đo Hit@3 và Hit@5, kèm danh sách câu bị trượt. Đồng thời đo thêm biến thể câu hỏi RÚT GỌN
(bỏ cụm "của Đại học Kỹ thuật Sông Hàn" vì cụm này có trong mọi tài liệu nên có thể gây nhiễu).

Cách dùng (đặt cạnh chatbot.py, trong data/ chỉ để 28 file DOC001..DOC028.pdf):
    python compare_embeddings.py                       # so sánh minilm, minilm-norm, e5-small, e5-base
    set COMPARE_KEYS=minilm,e5-small,bge-m3            # (cmd) hoặc  $env:COMPARE_KEYS="..."  (PowerShell)
    python compare_embeddings.py

Lần đầu mỗi model sẽ được tải về nên chờ lâu hơn. Nếu máy yếu, chạy từng model một bằng COMPARE_KEYS.
Kết quả lưu vào compare_embeddings.csv.
"""
import csv
import os
import time

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embeddings_config import get_embeddings, MODELS
from evaluate_retrieval import TESTS, norm, DATA_DIR

KEYS = [k.strip() for k in os.environ.get("COMPARE_KEYS", "minilm,minilm-norm,e5-small,e5-base").split(",") if k.strip()]
SCHOOL = " của Đại học Kỹ thuật Sông Hàn"


def misses(store, k, shorten):
    out = []
    for fid, question, answer in TESTS:
        q = question.replace(SCHOOL, "") if shorten else question
        results = store.similarity_search(q, k=k)
        if not any(norm(answer).lower() in norm(d.page_content).lower() for d in results):
            out.append(fid)
    return out


def main():
    docs = PyPDFDirectoryLoader(DATA_DIR).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    n = len(TESTS)
    print(f"{len({d.metadata.get('source') for d in docs})} file, {len(splits)} đoạn, {n} câu hỏi.\n")

    rows = []
    for key in KEYS:
        if key not in MODELS:
            print(f"Bỏ qua '{key}': không có trong embeddings_config.MODELS")
            continue
        t0 = time.time()
        try:
            store = FAISS.from_documents(splits, get_embeddings(key))
        except Exception as e:
            print(f"[{key}] không dựng được ({e}). Bỏ qua.\n")
            continue
        res = {}
        for shorten in (False, True):
            for k in (3, 5):
                res[(shorten, k)] = misses(store, k, shorten)
        dt = time.time() - t0
        rows.append([key, MODELS[key]["name"],
                     f"{(n - len(res[(False, 3)])) / n:.0%}", f"{(n - len(res[(False, 5)])) / n:.0%}",
                     f"{(n - len(res[(True, 3)])) / n:.0%}", f"{(n - len(res[(True, 5)])) / n:.0%}",
                     " ".join(res[(False, 3)]) or "-", " ".join(res[(False, 5)]) or "-", f"{dt:.0f}s"])
        print(f"[{key}] xong trong {dt:.0f}s")

    if not rows:
        raise SystemExit("Không có model nào chạy được.")

    head = ["model", "ten_day_du", "Hit@3", "Hit@5", "Hit@3_rut_gon", "Hit@5_rut_gon", "tuot@3", "tuot@5", "thoi_gian"]
    print("\n" + " | ".join(head[:1] + head[2:]))
    for r in rows:
        print(" | ".join([r[0]] + r[2:]))
    with open("compare_embeddings.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print("\nĐã lưu: compare_embeddings.csv")
    print("Chọn model có Hit@3 cao và ít câu trượt; nhớ còn phải cân nhắc độ nặng của model so với máy của nhóm.")


if __name__ == "__main__":
    main()
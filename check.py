import os
import requests
from dotenv import load_dotenv

load_dotenv()  # đọc file .env
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("Chưa có GOOGLE_API_KEY trong file .env")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print("Đang kết nối tới Google AI Studio...")
response = requests.get(url)

if response.status_code == 200:
    models = response.json().get("models", [])
    print("\n✅ THÀNH CÔNG! API Key của bạn hỗ trợ các model sau:")
    for m in models:
        # Chỉ in ra các model hỗ trợ tạo văn bản
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(f"- {m['name'].replace('models/', '')}")
else:
    print("\n❌ LỖI TRUY CẬP:")
    print(response.text)
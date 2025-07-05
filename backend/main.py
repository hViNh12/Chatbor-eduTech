import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# === KEY GOOGLE GEMINI (viết thẳng vào code – chỉ dùng tạm) ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyB6r5xzqAnTgVWcWwJJDXbmLHFdbIp5rTo"

# === Models & Config ===
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
app = FastAPI(title="EMG EduBot API")

# === Crawl EMG website ===
def crawl_emg_page(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else url
        text = " ".join(p.get_text() for p in soup.find_all(["p", "li", "h1", "h2", "h3"])).replace("\xa0", " ").strip()
        return Document(page_content=text, metadata={"url": url, "title": title})
    except:
        return None

emg_urls = [
    "https://emg.vn/",
    "https://emg.vn/gioi-thieu/",
    "https://emg.vn/chuong-trinh/",
    "https://emg.vn/lien-he/",
    "https://emg.vn/tin-tuc/",
    "https://emg.vn/category/tuyen-sinh/",
    "https://emg.vn/category/bai-viet-noi-bat/"
]

emg_docs = [doc for url in emg_urls if (doc := crawl_emg_page(url))]
manual_knowledge = [
    Document(page_content="Vốn điều lệ của EMG Education là khoảng 100 tỷ đồng."),
    Document(page_content="Người đại diện pháp luật là ông Trần Minh Sơn. Vợ ông là bà Trần Thị Lệ."),
    Document(page_content="Học phí tại EMG dao động từ 40 triệu đến 100 triệu mỗi năm."),
    Document(page_content="EMG triển khai chương trình Cambridge tại Việt Nam cho học sinh THCS."),
]
emg_docs.extend(manual_knowledge)

# === Split documents ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = []
for doc in emg_docs:
    chunks = splitter.create_documents([doc.page_content])
    split_docs.extend(chunks)

emg_vectorstore = FAISS.from_documents(split_docs, embedding_model)

# === Prompt ===
prompt_template = PromptTemplate(
    template="""
Bạn là trợ lý học tập EMG EduBot. Dưới đây là hồ sơ của học sinh và tài liệu EMG.

--- Hồ sơ người học ---
{profile}

--- Tài liệu liên quan ---
{context}

❓ Câu hỏi: {question}
""",
    input_variables=["profile", "context", "question"]
)

# === Request model ===
class QueryRequest(BaseModel):
    question: str
    user_profile: str

# === Endpoint ===
@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.question
    profile = request.user_profile

    profile_doc = Document(page_content=profile)
    profile_vector = FAISS.from_documents([profile_doc], embedding_model)

    retriever = emg_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    emg_docs = retriever.invoke(question)
    profile_docs = profile_vector.similarity_search(question, k=2)

    context_text = "\n\n".join(doc.page_content for doc in emg_docs + profile_docs)

    final_prompt = prompt_template.invoke({
        "profile": profile,
        "context": context_text,
        "question": question
    })

    response = llm.invoke(final_prompt)
    return {"answer": response.content.strip()}

# === Run locally ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)

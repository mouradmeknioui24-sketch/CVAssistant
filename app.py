import streamlit as st
from dotenv import load_dotenv
import json, re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI CV Assistant",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# STYLES (SUBTLE + MOBILE SAFE)
# --------------------------------------------------
st.markdown("""
<style>
.upload-hint {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 0.25rem;
}
.chat-hr {
    background-color:#F0F0F0;
    padding:12px;
    border-radius:12px;
    margin-bottom:6px;
    color:#000;
}
.chat-candidate {
    background-color:#E6E6FA;
    padding:12px;
    border-radius:12px;
    margin-bottom:12px;
    color:#000;
}
.match-box {
    background-color:#ECFDF5;
    border-left:5px solid #10B981;
    padding:12px;
    border-radius:8px;
    margin-top:12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def extract_cv_profile(cv_text):
    prompt = f"""
Extract a structured JSON profile from the CV text.
Return ONLY valid JSON.

Fields:
- name
- title
- summary
- skills
- experience
- education

CV:
{cv_text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)

def build_vectorstore(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return Chroma.from_texts(chunks, embeddings)

def analyze_match(cv_text, jd_text):
    prompt = f"""
You are an expert technical recruiter.

1. Give a match score from 0 to 100.
2. Give a SHORT reason (1–2 sentences).

Return JSON:
{{"score": number, "reason": string}}

CV:
{cv_text}

JOB:
{jd_text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)

def answer_hr_question(question, retriever, jd_text):
    docs = retriever.similarity_search(question, k=5)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
You are the candidate speaking as yourself.
Answer clearly, professionally, and truthfully.

CV CONTEXT:
{context}

JOB CONTEXT:
{jd_text}

QUESTION:
{question}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🧠 AI CV Assistant")
st.caption("Upload a CV, match it to a job, and interview the candidate through AI")

st.markdown("### 📂 Upload Documents")

# ---- CV Upload
st.markdown("<div class='upload-hint'>⬆️ Upload candidate CV (PDF)</div>", unsafe_allow_html=True)
uploaded_cv = st.file_uploader(
    "Upload CV",
    type=["pdf"],
    label_visibility="collapsed"
)

# ---- JD Upload
st.markdown("<div class='upload-hint'>⬆️ Upload job description (PDF or TXT)</div>", unsafe_allow_html=True)
uploaded_jd = st.file_uploader(
    "Upload Job Description",
    type=["pdf", "txt"],
    label_visibility="collapsed"
)

# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------
if uploaded_cv and uploaded_jd:
    with st.spinner("Analyzing CV and Job Description..."):
        cv_text = read_pdf(uploaded_cv)

        if uploaded_jd.type == "application/pdf":
            jd_text = read_pdf(uploaded_jd)
        else:
            jd_text = uploaded_jd.read().decode("utf-8")

        # Silent CV → JSON (not shown)
        cv_profile = extract_cv_profile(cv_text)

        # Vector store
        vectorstore = build_vectorstore(cv_text)
        retriever = vectorstore.as_retriever()

        # Match analysis
        match = analyze_match(cv_text, jd_text)

    # ---- MATCH RESULT
    st.markdown(
        f"""
        <div class="match-box">
        <b>Match Score:</b> {match["score"]}%<br/>
        <b>Reason:</b> {match["reason"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💬 HR Interview")

    question = st.text_input(
        "Ask a question to the candidate",
        placeholder="Why are you a good fit for this role?"
    )

    if question:
        answer = answer_hr_question(question, retriever, jd_text)
        st.session_state.chat_history.append((question, answer))

    # ---- CHAT HISTORY
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(
            f"<div class='chat-hr'><b>HR:</b> {q}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='chat-candidate'><b>Candidate:</b> {a}</div>",
            unsafe_allow_html=True
        )

else:
    st.info("Upload both a CV and a Job Description to begin.")
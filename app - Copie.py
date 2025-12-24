import streamlit as st
from dotenv import load_dotenv
import json, re, os
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="🤖 AI CV Assistant | FindReward", layout="centered")

st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#4B0082; margin-bottom:4px;">🤖 AI CV Assistant</h1>
    <div style="font-size:14px; color:#6B7280;">
        Interview a candidate through their AI twin
    </div>
    <div style="margin-top:6px;font-size:12px;font-weight:600;color:#047857;letter-spacing:0.08em;">
        POWERED BY <a href="https://findreward.net" target="_blank"
        style="color:#047857; text-decoration:none;">FindReward.net</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# MODELS
# --------------------------------------------------
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not set in Streamlit Secrets")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key, default in {
    "cv_profile": None,
    "job_profile": None,
    "match_analysis": None,
    "vectorstore": None,
    "chat_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------
# STYLES
# --------------------------------------------------
st.markdown("""
<style>
.upload-hint { font-size:0.85rem; color:#6b7280; margin-bottom:4px; }
.chat-hr { background:#F3F4F6; padding:12px; border-radius:12px; margin-bottom:6px; color:#111827; }
.chat-candidate { background:#ECFDF5; padding:12px; border-radius:12px; margin-bottom:14px; color:#064E3B; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def safe_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group(0))

def extract_pdf(file):
    reader = PdfReader(file)
    docs = []
    for i, page in enumerate(reader.pages):
        if page.extract_text():
            docs.append(Document(page_content=page.extract_text(), metadata={"page": i+1}))
    return docs

def parse_cv(docs):
    text = "\n\n".join(d.page_content for d in docs)
    return safe_json(llm.invoke([HumanMessage(content=f"Return ONLY JSON CV:\n{text}")]).content)

def parse_job(text):
    return safe_json(llm.invoke([HumanMessage(content=f"Return ONLY JSON Job:\n{text}")]).content)

def match_cv_job(cv, job):
    return safe_json(llm.invoke([HumanMessage(content=f"""
Return ONLY JSON:
{{"match_score":0-100,"reason":"short explanation"}}

CV:{json.dumps(cv)}
JOB:{json.dumps(job)}
""")]).content)

def extract_job_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = "\n".join(
        line.strip()
        for line in soup.get_text().splitlines()
        if len(line.strip()) > 30
    )

    if len(text) < 500:
        raise ValueError("Extracted job content too short")

    return text

# --------------------------------------------------
# UPLOAD UI
# --------------------------------------------------
st.markdown("### 📂 Upload Documents")

st.markdown("<div class='upload-hint'>⬆️ Upload candidate CV (PDF)</div>", unsafe_allow_html=True)
uploaded_cv = st.file_uploader("CV", type=["pdf"], label_visibility="collapsed")

st.markdown("<div class='upload-hint'>⬆️ Upload job description (PDF or TXT)</div>", unsafe_allow_html=True)
uploaded_jd = st.file_uploader("JD", type=["pdf", "txt"], label_visibility="collapsed")

st.markdown("""
<div class='upload-hint'>
🔗 Or paste job offer link (Indeed, ATS, career page)
</div>
""", unsafe_allow_html=True)

job_url = st.text_input(
    "",
    placeholder="https://www.indeed.com/viewjob?jk=...",
    label_visibility="collapsed"
)

st.info("If a job URL is provided, it will be used instead of the uploaded job description.")

# --------------------------------------------------
# PROCESS CV
# --------------------------------------------------
if uploaded_cv and not st.session_state.cv_profile:
    with st.spinner("Processing CV..."):
        cv_docs = extract_pdf(uploaded_cv)
        st.session_state.cv_profile = parse_cv(cv_docs)

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(cv_docs)
        st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)

    st.success("✅ CV processed")

# --------------------------------------------------
# PROCESS JOB (URL > FILE)
# --------------------------------------------------
if st.session_state.cv_profile and not st.session_state.job_profile:
    try:
        if job_url:
            with st.spinner("Fetching job from URL..."):
                job_text = extract_job_from_url(job_url)
                st.session_state.job_profile = parse_job(job_text)
            st.success("✅ Job extracted from URL")

        elif uploaded_jd:
            with st.spinner("Processing Job Description..."):
                if uploaded_jd.type == "application/pdf":
                    jd_docs = extract_pdf(uploaded_jd)
                    job_text = "\n\n".join(d.page_content for d in jd_docs)
                else:
                    job_text = uploaded_jd.read().decode("utf-8")

                st.session_state.job_profile = parse_job(job_text)
            st.success("✅ Job Description processed")

    except Exception as e:
        st.error("❌ Unable to extract job information")

# --------------------------------------------------
# MATCH ANALYSIS (PRO CARD)
# --------------------------------------------------
if st.session_state.cv_profile and st.session_state.job_profile and not st.session_state.match_analysis:
    with st.spinner("Analyzing match..."):
        st.session_state.match_analysis = match_cv_job(
            st.session_state.cv_profile,
            st.session_state.job_profile
        )

if st.session_state.match_analysis:
    score = st.session_state.match_analysis["match_score"]
    reason = st.session_state.match_analysis["reason"]

    st.markdown(f"""
    <div style="background:#F9FAFB;border-left:6px solid #10B981;
    padding:20px;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.08);">
        <div style="display:flex;justify-content:space-between;">
            <div style="font-size:14px;font-weight:600;color:#065F46;">MATCH SCORE</div>
            <div style="font-size:32px;font-weight:800;color:#047857;">{score}%</div>
        </div>
        <div style="height:8px;background:#D1FAE5;border-radius:4px;margin:10px 0;">
            <div style="width:{score}%;height:100%;
            background:linear-gradient(90deg,#10B981,#22D3EE);"></div>
        </div>
        <div style="font-size:14px;color:#111827;">
            <b>Why this match:</b><br/>{reason}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# INTERVIEW
# --------------------------------------------------
st.markdown("---")
st.subheader("🎤 HR Interview")

if st.session_state.vectorstore:
    question = st.text_input("Ask a question", placeholder="Why are you a good fit for this role?")
    if st.button("Ask") and question:
        retriever = st.session_state.vectorstore.as_retriever(k=4)
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        answer = llm.invoke([
            SystemMessage(content=f"""
You are the candidate speaking in FIRST PERSON.
Job:{json.dumps(st.session_state.job_profile)}
Match:{json.dumps(st.session_state.match_analysis)}
Context:{context}
"""),
            HumanMessage(content=question)
        ]).content

        st.session_state.chat_history.append((question, answer))

# --------------------------------------------------
# CHAT DISPLAY
# --------------------------------------------------
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"<div class='chat-hr'><b>HR:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-candidate'><b>Candidate:</b> {a}</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<hr/>
<div style="text-align:center;font-size:12px;color:#6B7280;">
© 2025 <a href="https://findreward.net" target="_blank"
style="color:#047857;text-decoration:none;">FindReward.net</a> — All rights reserved
</div>
""", unsafe_allow_html=True)

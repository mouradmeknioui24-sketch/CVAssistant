import streamlit as st
from dotenv import load_dotenv
import json, re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
import os
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
    <div style="
        font-size:15px;
        color:#6B7280;
        max-width:520px;
        margin:0 auto 6px auto;
        line-height:1.6;
    ">
        Analyze your CV, compare it to real job roles, and practice interviews
        with your AI twin — before you apply.
    </div>
    <div style="font-size:14px; color:#6B7280;">
        Interview a candidate through their AI twin
    </div>
    <div style="
        margin-top:6px;
        font-size:12px;
        font-weight:600;
        color:#047857;
        letter-spacing:0.08em;
    ">
        POWERED BY <a href="https://findreward.net" target="_blank"
        style="color:#047857; text-decoration:none;">FindReward.net</a>
    </div>
    <div style="
        font-size:13px;
        color:#374151;
        max-width:520px;
        margin:0 auto 8px auto;
        line-height:1.5;
    ">
        Built for job seekers and recruiters to simulate realistic hiring
        conversations before the first interview.
    </div>
</div>
<div style="display:none;">
    <script type="text/javascript">
        (function(c,l,a,r,i,t,y){
            c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        })(window, document, "clarity", "script", "uqq6o9ppuj");
    </script>
</div>
<div style="display:none;">
<script async defer src="https://tools.luckyorange.com/core/lo.js?site-id=83e00574"></script>
</div>
""", unsafe_allow_html=True)

# ==================================================

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
.upload-hint {
    font-size:0.85rem;
    color:#6b7280;
    margin-bottom:0.25rem;
}
.chat-hr {
    background:#F3F4F6;
    padding:12px;
    border-radius:12px;
    margin-bottom:6px;
    color:#111827;
}
.chat-candidate {
    background:#ECFDF5;
    padding:12px;
    border-radius:12px;
    margin-bottom:14px;
    color:#064E3B;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def safe_json(text):
    if not text or not text.strip():
        st.error("LLM returned empty output. Try again.")
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        st.error("Could not find valid JSON in LLM output.")
        st.error(text)
        return {}
    try:
        return json.loads(match.group(0))
    except Exception as e:
        st.error(e)
        return {}

def extract_pdf(file):
    reader = PdfReader(file)
    docs = []
    for i, page in enumerate(reader.pages):
        if page.extract_text():
            docs.append(Document(page_content=page.extract_text(), metadata={"page": i+1}))
    return docs

def parse_cv(docs):
    text = "\n\n".join(d.page_content for d in docs)
    return safe_json(llm.invoke([HumanMessage(content=f"Return ONLY valid JSON with CV info:\n{text}")]).content)

def parse_job(text):
    return safe_json(llm.invoke([HumanMessage(content=f"Return ONLY valid JSON with Job info:\n{text}")]).content)

def match_cv_job(cv, job):
    prompt = f"""Return ONLY JSON:
{{"match_score":0-100,"reason":"short explanation"}}
CV: {json.dumps(cv)}
JOB: {json.dumps(job)}"""
    return safe_json(llm.invoke([HumanMessage(content=prompt)]).content)

# --------------------------------------------------
# FETCH JOB TEXT FROM URL
# --------------------------------------------------
def fetch_job_text_from_link(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)
    except Exception as e:
        return str(e)

# --------------------------------------------------
# UPLOAD CV
# --------------------------------------------------
st.markdown("### 📂 Candidate CV")
st.markdown("<div class='upload-hint'>⬆️ Upload candidate CV (PDF)</div>", unsafe_allow_html=True)
uploaded_cv = st.file_uploader("CV", type=["pdf"], label_visibility="collapsed")

if uploaded_cv and not st.session_state.cv_profile:
    with st.spinner("Processing CV..."):
        cv_docs = extract_pdf(uploaded_cv)
        st.session_state.cv_profile = parse_cv(cv_docs)
        chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(cv_docs)
        st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
    st.success("✅ CV processed")

# --------------------------------------------------
# JOB DESCRIPTION OPTIONS
# --------------------------------------------------
st.markdown("### 📄 Job Description Input")
job_option = st.radio(
    "Choose how to provide the Job Description:",
    ("Upload File", "Provide Job Link", "Paste Text")
)

if job_option == "Upload File":
    uploaded_jd = st.file_uploader("Job Description", type=["pdf", "txt"], label_visibility="collapsed")
    if uploaded_jd and not st.session_state.job_profile:
        with st.spinner("Processing Job Description..."):
            if uploaded_jd.type == "application/pdf":
                jd_docs = extract_pdf(uploaded_jd)
                jd_text = "\n\n".join(d.page_content for d in jd_docs)
            else:
                jd_text = uploaded_jd.read().decode("utf-8")
            st.session_state.job_profile = parse_job(jd_text)
        st.success("✅ Job Description processed")

elif job_option == "Provide Job Link":
    job_link = st.text_input("Paste job link here")
    if job_link and not st.session_state.job_profile:
        with st.spinner("Fetching job page..."):
            st.session_state.job_profile = parse_job(fetch_job_text_from_link(job_link))
        st.success("✅ Job Description processed")

elif job_option == "Paste Text":
    jd_text = st.text_area("Paste the job description here")
    if jd_text and not st.session_state.job_profile:
        with st.spinner("Processing job text..."):
            st.session_state.job_profile = parse_job(jd_text)
        st.success("✅ Job Description processed")

# --------------------------------------------------
# MATCH ANALYSIS + INTERVIEW + FOOTER
# (UNCHANGED)
# --------------------------------------------------

i am giving you my code inject this in my code and please change nothing else




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
    """Try to extract JSON from LLM output robustly."""
    if not text or not text.strip():
        st.error("LLM returned empty output. Try again.")
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        st.error("Could not find valid JSON in LLM output. Please check your input.")
        st.error(f"Raw output:\n{text}")
        return {}
    try:
        return json.loads(match.group(0))
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        st.error(f"Raw JSON string:\n{match.group(0)}")
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
    prompt = f"Return ONLY valid JSON with CV info:\n{text}"
    return safe_json(llm.invoke([HumanMessage(content=prompt)]).content)

def parse_job(text):
    prompt = f"Return ONLY valid JSON with Job info:\n{text}"
    return safe_json(llm.invoke([HumanMessage(content=prompt)]).content)

def match_cv_job(cv, job):
    prompt = f"""
Return ONLY JSON:
{{"match_score":0-100,"reason":"short explanation"}}

CV: {json.dumps(cv)}
JOB: {json.dumps(job)}
"""
    return safe_json(llm.invoke([HumanMessage(content=prompt)]).content)

# --------------------------------------------------
# FETCH JOB TEXT FROM URL
# --------------------------------------------------
def fetch_job_text_from_link(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"Could not fetch page content: {e}"

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

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(cv_docs)
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

jd_text = None

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
    job_link = st.text_input("Paste job link here (Indeed, LinkedIn, etc.)")
    if job_link and not st.session_state.job_profile:
        with st.spinner("Fetching and processing job link..."):
            jd_text = fetch_job_text_from_link(job_link)
            st.session_state.job_profile = parse_job(jd_text)
        st.success("✅ Job Description processed from link")

elif job_option == "Paste Text":
    jd_text_input = st.text_area("Paste the job description here")
    if jd_text_input and not st.session_state.job_profile:
        with st.spinner("Processing pasted job description..."):
            jd_text = jd_text_input
            st.session_state.job_profile = parse_job(jd_text)
        st.success("✅ Job Description processed from pasted text")

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

    st.markdown(
        f"""
        <div style="
            background-color:#2F3632;
            border-left:6px solid #10B981;
            padding:14px;
            border-radius:10px;
            color:#F7F5F5;
            margin-top:12px;
        ">
            <div style="font-size:26px; font-weight:700;">
                Match Score: {score}%
            </div>
            <div style="margin-top:6px; font-size:14px;">
                <b>Reason:</b> {reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# HR INTERVIEW
# --------------------------------------------------
st.markdown("---")
st.subheader("🎤 HR Interview")

if st.session_state.vectorstore:
    question = st.text_input("Ask a question", placeholder="Why are you a good fit for this role?")
    if st.button("Ask") and question:
        retriever = st.session_state.vectorstore.as_retriever(k=4)
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are the candidate speaking in FIRST PERSON.

Job Info:
{json.dumps(st.session_state.job_profile)}

Match Info:
{json.dumps(st.session_state.match_analysis)}

Context:
{context}

Answer clearly and professionally.
"""
        answer = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=question)]).content
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
<div style="text-align:center; font-size:12px; color:#6B7280;">
    © 2025 <a href="https://findreward.net" target="_blank"
    style="color:#047857; text-decoration:none;">FindReward.net</a> — All rights reserved
</div>
""", unsafe_allow_html=True)

st.markdown("""
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "uqq6o9ppuj");
</script>
""", unsafe_allow_html=True)
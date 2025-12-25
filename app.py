import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import json, re, os, requests
from pypdf import PdfReader
from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="🤖 AI CV Assistant | FindReward", layout="centered")

# --------------------------------------------------
# ANALYTICS (LOAD ONCE)
# --------------------------------------------------
if "analytics_loaded" not in st.session_state:
    components.html(
        """
        <script type="text/javascript">
        (function(c,l,a,r,i,t,y){
            c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
            t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
            y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
        })(window, document, "clarity", "script", "uqq6o9ppuj");
        </script>

        <script async defer src="https://tools.luckyorange.com/core/lo.js?site-id=83e00574"></script>
        """,
        height=0,
    )
    st.session_state.analytics_loaded = True

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#4B0082;">🤖 AI CV Assistant</h1>
    <div style="font-size:15px; color:#6B7280; max-width:520px; margin:auto;">
        Analyze your CV, compare it to real job roles, and practice interviews
        with your AI twin — before you apply.
    </div>
    <div style="margin-top:6px; font-size:12px; font-weight:600; color:#047857;">
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
    st.error("❌ OPENAI_API_KEY not set")
    st.stop()

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key in [
    "cv_profile",
    "job_profile",
    "match_analysis",
    "vectorstore",
    "chat_history",
]:
    st.session_state.setdefault(key, None if key != "chat_history" else [])

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def safe_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}

def extract_pdf(file):
    reader = PdfReader(file)
    return [
        Document(page_content=p.extract_text(), metadata={"page": i})
        for i, p in enumerate(reader.pages)
        if p.extract_text()
    ]

def parse_cv(docs):
    text = "\n\n".join(d.page_content for d in docs)
    return safe_json(llm.invoke([HumanMessage(content=f"Return ONLY JSON CV:\n{text}")]).content)

def parse_job(text):
    return safe_json(llm.invoke([HumanMessage(content=f"Return ONLY JSON Job:\n{text}")]).content)

def match_cv_job(cv, job):
    prompt = f"""
Return ONLY JSON:
{{"match_score":0-100,"reason":"short explanation"}}
CV: {json.dumps(cv)}
JOB: {json.dumps(job)}
"""
    return safe_json(llm.invoke([HumanMessage(content=prompt)]).content)

def fetch_job_text_from_link(url):
    soup = BeautifulSoup(requests.get(url, timeout=10).text, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    return "\n".join(l.strip() for l in soup.get_text().splitlines() if l.strip())

# --------------------------------------------------
# CV UPLOAD
# --------------------------------------------------
st.markdown("### 📂 Candidate CV")
uploaded_cv = st.file_uploader("Upload CV", type=["pdf"])

if uploaded_cv:
    cv_docs = extract_pdf(uploaded_cv)
    st.session_state.cv_profile = parse_cv(cv_docs)
    chunks = CharacterTextSplitter(1000, 200).split_documents(cv_docs)
    st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
    st.success("✅ CV processed")

# --------------------------------------------------
# JOB INPUT
# --------------------------------------------------
st.markdown("### 📄 Job Description")
job_option = st.radio("Choose input:", ["Upload File", "Job Link", "Paste Text"])

if job_option == "Upload File":
    jd = st.file_uploader("Upload JD", type=["pdf", "txt"])
    if jd:
        text = jd.read().decode() if jd.type == "text/plain" else "\n".join(
            d.page_content for d in extract_pdf(jd)
        )
        st.session_state.job_profile = parse_job(text)
        st.success("✅ Job processed")

elif job_option == "Job Link":
    link = st.text_input("Paste job URL")
    if link:
        st.session_state.job_profile = parse_job(fetch_job_text_from_link(link))
        st.success("✅ Job processed")

else:
    text = st.text_area("Paste job description")
    if text:
        st.session_state.job_profile = parse_job(text)
        st.success("✅ Job processed")

# --------------------------------------------------
# MATCH ANALYSIS
# --------------------------------------------------
if st.session_state.cv_profile and st.session_state.job_profile:
    if not st.session_state.match_analysis:
        st.session_state.match_analysis = match_cv_job(
            st.session_state.cv_profile,
            st.session_state.job_profile,
        )

    m = st.session_state.match_analysis
    st.markdown(f"""
    <div style="background:#2F3632;padding:16px;border-radius:10px;color:white;">
        <div style="font-size:26px;font-weight:700;">Match Score: {m["match_score"]}%</div>
        <div style="margin-top:8px;font-size:14px;"><b>Reason:</b> {m["reason"]}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# INTERVIEW
# --------------------------------------------------
if st.session_state.match_analysis:
    st.markdown("---")
    st.subheader("🎤 HR Interview")

    q = st.text_input("Ask a question")
    if st.button("Ask") and q:
        docs = st.session_state.vectorstore.as_retriever(k=4).invoke(q)
        ctx = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are the candidate speaking in FIRST PERSON.

Job: {json.dumps(st.session_state.job_profile)}
Match: {json.dumps(st.session_state.match_analysis)}

Context:
{ctx}
"""
        a = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=q)]).content
        st.session_state.chat_history.append((q, a))

    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**HR:** {q}")
        st.markdown(f"**Candidate:** {a}")

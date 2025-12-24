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
st.set_page_config(page_title="🤖 AI CV Assistant", layout="centered")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🤖 AI CV Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Interview a candidate through their AI twin</p>", unsafe_allow_html=True)
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
.upload-hint { font-size:0.85rem; color:#6b7280; margin-bottom:0.25rem; }
.chat-hr { background:#F0F0F0; padding:12px; border-radius:12px; margin-bottom:5px; color:#000; }
.chat-candidate { background:#E6E6FA; padding:12px; border-radius:12px; margin-bottom:12px; color:#000; }
.match-box { background:#ECFDF5; border-left:5px solid #10B981; padding:12px; border-radius:8px; }
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
# UPLOAD UI
# --------------------------------------------------
st.markdown("### 📂 Upload Documents")

st.markdown("<div class='upload-hint'>⬆️ Upload candidate CV (PDF)</div>", unsafe_allow_html=True)
uploaded_cv = st.file_uploader("CV", type=["pdf"], label_visibility="collapsed")

st.markdown("<div class='upload-hint'>⬆️ Upload job description (PDF or TXT)</div>", unsafe_allow_html=True)
uploaded_jd = st.file_uploader("JD", type=["pdf", "txt"], label_visibility="collapsed")

# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------
if uploaded_cv and not st.session_state.cv_profile:
    with st.spinner("Processing CV..."):
        cv_docs = extract_pdf(uploaded_cv)
        st.session_state.cv_profile = parse_cv(cv_docs)

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(cv_docs)
        st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)

    st.success("✅ CV processed")

if uploaded_jd and not st.session_state.job_profile:
    with st.spinner("Processing Job Description..."):
        if uploaded_jd.type == "application/pdf":
            jd_docs = extract_pdf(uploaded_jd)
            jd_text = "\n\n".join(d.page_content for d in jd_docs)
        else:
            jd_text = uploaded_jd.read().decode("utf-8")

        st.session_state.job_profile = parse_job(jd_text)

    st.success("✅ Job Description processed")

# --------------------------------------------------
# MATCH ANALYSIS
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
        background: linear-gradient(135deg, #F0FDF4, #ECFEFF);
        border: 1px solid #D1FAE5;
        padding: 18px;
        border-radius: 14px;
        margin-top: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        color: #064E3B;
     ">

        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        ">
            <div style="
                font-size: 14px;
                letter-spacing: 0.08em;
                font-weight: 600;
                color:#047857;
            ">
                MATCH SCORE
            </div>

            <div style="
                font-size: 28px;
                font-weight: 800;
                color: #065F46;
            ">
                {score}%
            </div>
        </div>

        <div style="
            height: 6px;
            width: 100%;
            background-color: #D1FAE5;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 12px;
        ">
            <div style="
                width: {score}%;
                height: 100%;
                background: linear-gradient(90deg, #10B981, #22D3EE);
            "></div>
        </div>

        <div style="
            font-size: 14px;
            line-height: 1.6;
            color: #064E3B;
        ">
            <b>Why this match:</b><br/>
            {reason}
        </div>

     </div>
     """,
    unsafe_allow_html=True
)



# --------------------------------------------------
# INTERVIEW
# --------------------------------------------------
st.markdown("---")
st.subheader("🎤 HR Interview")

if not st.session_state.cv_profile:
    st.info("Upload a CV to start the interview.")
else:
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
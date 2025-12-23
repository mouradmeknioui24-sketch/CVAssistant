import streamlit as st
from dotenv import load_dotenv
import json, re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# -------------------- ENV & MODELS --------------------
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🤖 AI CV Assistant", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🤖 AI CV Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Upload CV & Job Description, see matching score, and interview the candidate.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SESSION STATE --------------------
for key in ["cv_profile", "vectorstore", "chat_history", "job_profile", "match_analysis"]:
    if key not in st.session_state or st.session_state[key] is None:
        st.session_state[key] = [] if key=="chat_history" else None

# -------------------- HELPERS --------------------
def safe_json_load(text: str) -> dict:
    if not text or not text.strip(): raise ValueError("Empty LLM output")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: raise ValueError("No JSON found in output")
    return json.loads(match.group(0))

def extract_text_from_pdf(file) -> list[Document]:
    reader = PdfReader(file)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"page": i+1}))
    return docs

# -------------------- SIDEBAR UPLOAD --------------------
st.sidebar.header("📂 Upload Files")
uploaded_cv = st.sidebar.file_uploader("Upload Candidate CV (PDF)", type=["pdf"])
uploaded_jd = st.sidebar.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf","txt"])

# -------------------- PROCESS CV --------------------
if uploaded_cv and st.session_state.cv_profile is None:
    with st.spinner("Processing CV..."):
        pages = extract_text_from_pdf(uploaded_cv)
        full_text = "\n\n".join(p.page_content for p in pages)
        prompt = f"Return only JSON with CV info. CV TEXT: {full_text}"
        response = llm.invoke([SystemMessage(content="CV parser"), HumanMessage(content=prompt)])
        st.session_state.cv_profile = safe_json_load(response.content)

        # Vector store for semantic search
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        st.session_state.vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    st.success("✅ CV processed successfully")

# -------------------- PROCESS JOB --------------------
if uploaded_jd and st.session_state.job_profile is None:
    with st.spinner("Processing Job Description..."):
        if uploaded_jd.type == "application/pdf":
            jd_pages = extract_text_from_pdf(uploaded_jd)
            jd_text = "\n\n".join(p.page_content for p in jd_pages)
        else:
            jd_text = uploaded_jd.read().decode("utf-8")
        prompt = f"Return only JSON with Job info. JOB TEXT: {jd_text}"
        response = llm.invoke([SystemMessage(content="Job parser"), HumanMessage(content=prompt)])
        st.session_state.job_profile = safe_json_load(response.content)
    st.success("✅ Job Description processed successfully")

# -------------------- MATCH ANALYSIS --------------------
if st.session_state.cv_profile and st.session_state.job_profile and not st.session_state.match_analysis:
    with st.spinner("Analyzing CV-Job match..."):
        prompt = f"Return only JSON with match_score, strengths, missing_skills, reason. CV: {json.dumps(st.session_state.cv_profile)} JOB: {json.dumps(st.session_state.job_profile)}"
        response = llm.invoke([SystemMessage(content="Match analyzer"), HumanMessage(content=prompt)])
        st.session_state.match_analysis = safe_json_load(response.content)

# -------------------- DISPLAY MATCH --------------------
if st.session_state.match_analysis:
    score = st.session_state.match_analysis.get("match_score","N/A")
    reason = st.session_state.match_analysis.get("reason","No reason provided.")
    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown(f"<h2 style='color:#4B0082;'>{score}%</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#666;'>Preliminary Match Score</p>", unsafe_allow_html=True)
    with col2:
        st.info(f"**Reason:** {reason}")

st.markdown("---")

# -------------------- INTERVIEW MODE --------------------
if st.session_state.vectorstore and st.session_state.cv_profile:
    st.subheader("🎤 HR Interview Simulation")
    question = st.text_input("Ask a question to the candidate")
    if st.button("Ask") and question:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        system_prompt = f"""
You are the AI assistant of the CV owner.
Answer in FIRST PERSON.
Job Info: {json.dumps(st.session_state.job_profile, indent=2)}
Match Analysis: {json.dumps(st.session_state.match_analysis, indent=2)}
Rules: Highlight strengths, address gaps honestly, confident, no invented experience.
"""
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=f"Question: {question}\nContext:\n{context}")])
        st.session_state.chat_history.append((question, response.content))

# -------------------- DISPLAY CHAT --------------------
if st.session_state.chat_history:
    for q,a in reversed(st.session_state.chat_history):
        st.markdown(f"<div style='background-color:#F0F0F0;padding:10px;border-radius:8px'><b>HR:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#E6E6FA;padding:10px;border-radius:8px'><b>Candidate:</b> {a}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

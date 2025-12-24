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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🤖 AI CV Assistant", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🤖 AI CV Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Upload CV & Job Description for analysis</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- SECRETS --------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.warning("❌ OpenAI API key not set. Please set it in Streamlit Secrets.")
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

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

def extract_cv_profile(documents: list[Document]) -> dict:
    full_text = "\n\n".join(d.page_content for d in documents)
    prompt = f"Return only JSON with CV info. CV TEXT: {full_text}"
    response = llm.invoke([SystemMessage(content="CV parser"), HumanMessage(content=prompt)])
    return safe_json_load(response.content)

def extract_job_profile(text: str) -> dict:
    prompt = f"Return only JSON with Job info. JOB TEXT: {text}"
    response = llm.invoke([SystemMessage(content="Job parser"), HumanMessage(content=prompt)])
    return safe_json_load(response.content)

def analyze_match(cv: dict, job: dict) -> dict:
    prompt = f"Return only JSON with match_score (0-100), strengths, missing_skills, reason. CV: {json.dumps(cv)} JOB: {json.dumps(job)}"
    response = llm.invoke([SystemMessage(content="Match analyzer"), HumanMessage(content=prompt)])
    return safe_json_load(response.content)

# -------------------- FILE UPLOAD UI --------------------
st.markdown("### 📂 Upload Documents", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown(
        """
        <div style='background-color:#E6E6FA; padding:10px; border-radius:8px; text-align:center;'>
            <h4 style='color:#4B0082; margin:0;'>📂 Upload CV (PDF)</h4>
            <p style='color:#666; font-size:12px; margin:4px 0 0 0;'>Select candidate CV</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_cv = st.file_uploader(
        "",
        type=["pdf"],
        key="cv",
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div style='background-color:#FFF0F5; padding:10px; border-radius:8px; text-align:center; margin-top:10px;'>
            <h4 style='color:#4B0082; margin:0;'>📄 Upload JD</h4>
            <p style='color:#666; font-size:12px; margin:4px 0 0 0;'>PDF or TXT</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_jd = st.file_uploader(
        "",
        type=["pdf", "txt"],
        key="jd",
        label_visibility="collapsed"
    )


# -------------------- PROCESS CV --------------------
if uploaded_cv and st.session_state.cv_profile is None:
    with st.spinner("Processing CV..."):
        pages = extract_text_from_pdf(uploaded_cv)
        st.session_state.cv_profile = extract_cv_profile(pages)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        st.session_state.vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    st.success("✅ CV processed successfully")

# -------------------- PROCESS JOB --------------------
if uploaded_jd and st.session_state.job_profile is None:
    with st.spinner("Processing Job Description..."):
        if uploaded_jd.type == "application/pdf":
            jd_pages = extract_text_from_pdf(uploaded_jd)
            jd_text = "\n\n".join(d.page_content for p in jd_pages)
        else:
            jd_text = uploaded_jd.read().decode("utf-8")
        st.session_state.job_profile = extract_job_profile(jd_text)
    st.success("✅ Job Description processed successfully")

# -------------------- MATCH ANALYSIS --------------------
if st.session_state.cv_profile and st.session_state.job_profile and not st.session_state.match_analysis:
    with st.spinner("Analyzing CV-Job match..."):
        st.session_state.match_analysis = analyze_match(st.session_state.cv_profile, st.session_state.job_profile)

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

# -------------------- DISPLAY CHAT WITH BUBBLES --------------------
if st.session_state.chat_history:
    for q, a in reversed(st.session_state.chat_history):
        # HR question
        st.markdown(
            f"""
            <div style="
                background-color:#F0F0F0;
                padding:12px;
                border-radius:12px;
                margin-bottom:5px;
                color:#000;
            ">
            <b>HR:</b> {q}
            </div>
            """,
            unsafe_allow_html=True
        )
        # Candidate answer
        st.markdown(
            f"""
            <div style="
                background-color:#E6E6FA;
                padding:12px;
                border-radius:12px;
                margin-bottom:10px;
                color:#000;
            ">
            <b>Candidate:</b> {a}
            </div>
            """,
            unsafe_allow_html=True
        )
import os
from langchain.vectorstores import Chroma
PERSIST_DIR = "db/cv_store"
if os.path.exists(PERSIST_DIR):
    st.session_state.vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

# robust_imports.py  -- paste at top of your app

# standard libs
import os
import re
import shutil
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any, List
# Streamlit & dotenv
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# LangGraph (unchanged)
from langgraph.graph import StateGraph, START, END

# LangChain imports with fallbacks for different package versions
# ChatGoogleGenerativeAI: try common import locations used by LangChain + langchain-google-genai packages
ChatGoogleGenerativeAI = None
try:
    # newer unified langchain may expose chat models here
    from langchain.chat_models import ChatGoogleGenerativeAI
    ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
except Exception:
    try:
        # historical / separate package name
        from langchain_google_genai import ChatGoogleGenerativeAI
        ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
    except Exception:
        try:
            # alternate path some examples use
            from langchain_google_genai.chat import ChatGoogleGenerativeAI
            ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
        except Exception:
            ChatGoogleGenerativeAI = None

# RecursiveCharacterTextSplitter: try both langchain core and langchain-text-splitters package
RecursiveCharacterTextSplitter = None
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None

# Other LangChain pieces (with straightforward imports)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
except Exception:
    # fallback: older names / raise clear error later if used
    HuggingFaceEmbeddings = None
    FAISS = None
    PromptTemplate = None
    LLMChain = None

# sanity checks (fail fast with clear message)
missing = []
if ChatGoogleGenerativeAI is None:
    missing.append("ChatGoogleGenerativeAI (install langchain-google-genai or upgrade langchain)")
if RecursiveCharacterTextSplitter is None:
    missing.append("RecursiveCharacterTextSplitter (install langchain-text-splitters or upgrade langchain)")
if missing:
    # This will show up in Streamlit when you run and help debugging
    st.warning("Import warnings: " + "; ".join(missing))

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class YoutubeStateGraph(Dict):
    url:str
    video_id:str
    transcript:str
    chunks:List[str]
    vector_store:Any
    retriever:Any
    user_query:str
    llm_response:str

def fetch_youtube_video_id(graph:YoutubeStateGraph)->YoutubeStateGraph:
    video_id = ""
    url = graph["url"]
    for i in range(len(url)):
        if url[i:i+9]=="youtu.be/":
            video_id = url[i+9:i+20]
        elif url[i:i+8]=="watch?v=":
            video_id = url[i+8:i+19]
    print("Video ID:",video_id)
    return {"video_id":video_id}

def fetch_youtube_transcript(graph:YoutubeStateGraph)->YoutubeStateGraph:
    video_id = graph["video_id"]
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = ""
        for entry in transcript_list:
            transcript+=entry.text+" "
        return {"transcript":transcript}
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return {"transcript":""}

def spit_transcript(graph:YoutubeStateGraph)->YoutubeStateGraph:
    if graph["transcript"]=="":
        return {"chunks":[]}
    transcript = graph["transcript"]
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )
        chunks = splitter.split_text(transcript)
        return {"chunks":chunks}
    except Exception as e:
        print(f"Error during text splitting: {e}")
        return []

def create_embeddings(graph:YoutubeStateGraph)->YoutubeStateGraph:
    
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    chunks = graph["chunks"]
    VECTORSTORE_PATH = "faiss_index"

    print("Generating new embeddings...")
    vector_store = FAISS.from_texts(chunks, embeddings_model)

    vector_store.save_local(VECTORSTORE_PATH)
    print("FAISS index saved locally for future use!")

    return {"vector_store": vector_store}

def retriever(graph:YoutubeStateGraph)->YoutubeStateGraph:
    retriever = graph["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k":4})
    return {"retriever":retriever}

def augmentation(graph:YoutubeStateGraph)->YoutubeStateGraph:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2
        )
    context = graph["retriever"].invoke(graph["user_query"])
    template_string = """
    You are an AI assistant helping to answer user queries based on the context provided from a Youtube video transcript.
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not use your own knowledge or prior beliefs, only the context below.
    Context:
    {context}
    Question: {question}
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template_string
    )
    chain = prompt|llm
    response = chain.invoke({
        "context":context,
        "question":graph["user_query"]
    })
    
    response = response.content.strip()
    return {"llm_response":response}

# Initialize workflow
workflow = StateGraph(YoutubeStateGraph)
workflow.add_node("fetch_youtube_video_id", fetch_youtube_video_id)
workflow.add_node("fetch_youtube_transcript", fetch_youtube_transcript)
workflow.add_node("spit_transcript", spit_transcript)
workflow.add_node("create_embeddings", create_embeddings)
workflow.add_node("retriever", retriever)
workflow.add_node("augmentation", augmentation)
workflow.add_edge(START, "fetch_youtube_video_id")
workflow.add_edge("fetch_youtube_video_id", "fetch_youtube_transcript")
workflow.add_edge("fetch_youtube_transcript", "spit_transcript")
workflow.add_edge("spit_transcript", "create_embeddings")
workflow.add_edge("create_embeddings", "retriever")
workflow.add_edge("retriever", "augmentation")
workflow.add_edge("augmentation", END)

app = workflow.compile()

# Streamlit App
st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎥", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_url" not in st.session_state:
    st.session_state.current_url = None
if "workflow_state" not in st.session_state:
    st.session_state.workflow_state = None

# Sidebar for YouTube URL input
with st.sidebar:
    st.title("🎥 YouTube Video")
    st.markdown("---")
    
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a YouTube video URL to analyze"
    )
    
    if st.button("Load Video", type="primary", use_container_width=True):
        if youtube_url:
            # Check if URL is different from current
            if youtube_url != st.session_state.current_url:
                with st.spinner("Loading video transcript and creating embeddings..."):
                    try:
                        # Delete old FAISS index if it exists
                        VECTORSTORE_PATH = "faiss_index"
                        if os.path.exists(VECTORSTORE_PATH):
                            shutil.rmtree(VECTORSTORE_PATH)
                            st.info("🗑️ Deleted previous embeddings")
                        
                        # Process the video
                        inputs = {
                            "url": youtube_url,
                            "user_query": ""  # Placeholder, will be updated per query
                        }
                        
                        # Run workflow up to retriever setup
                        partial_state = app.invoke(inputs)
                        
                        # Store in session
                        st.session_state.current_url = youtube_url
                        st.session_state.workflow_state = partial_state
                        st.session_state.messages = []  # Clear chat history
                        
                        st.success("✅ Video loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading video: {str(e)}")
            else:
                st.info("This video is already loaded!")
        else:
            st.warning("Please enter a YouTube URL")
    
    if st.session_state.current_url:
        st.markdown("---")
        st.markdown("**Current Video:**")
        st.code(st.session_state.current_url, language=None)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main chat interface
st.title("💬 YouTube Video Chat Assistant")
st.markdown("Ask questions about the YouTube video content!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the video..."):
    if not st.session_state.current_url:
        st.warning("⚠️ Please load a YouTube video first using the sidebar!")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Update query in workflow state
                    inputs = {
                        "url": st.session_state.current_url,
                        "user_query": prompt
                    }
                    
                    # Run full workflow
                    final_state = app.invoke(inputs)
                    response = final_state["llm_response"]
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Show initial message if no video loaded
if not st.session_state.current_url:
    st.info("👈 Start by loading a YouTube video from the sidebar!")






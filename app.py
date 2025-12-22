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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None

# HuggingFaceEmbeddings - try langchain-community first
HuggingFaceEmbeddings = None
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

# FAISS - try langchain-community first
FAISS = None
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        FAISS = None

# PromptTemplate
PromptTemplate = None
try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None

# sanity checks (fail fast with clear message)
missing = []
if ChatGoogleGenerativeAI is None:
    missing.append("ChatGoogleGenerativeAI (install: pip install langchain-google-genai)")
if RecursiveCharacterTextSplitter is None:
    missing.append("RecursiveCharacterTextSplitter (install: pip install langchain-text-splitters)")
if HuggingFaceEmbeddings is None:
    missing.append("HuggingFaceEmbeddings (install: pip install langchain-community)")
if FAISS is None:
    missing.append("FAISS (install: pip install faiss-cpu langchain-community)")
if PromptTemplate is None:
    missing.append("PromptTemplate (install: pip install langchain-core)")

if missing:
    st.error("❌ Missing required dependencies:")
    for item in missing:
        st.error(f"  • {item}")
    st.stop()

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
    try:
        st.write("DEBUG: Entering fetch_youtube_video_id")
        video_id = ""
        url = graph["url"]
        for i in range(len(url)):
            if url[i:i+9]=="youtu.be/":
                video_id = url[i+9:i+20]
            elif url[i:i+8]=="watch?v=":
                video_id = url[i+8:i+19]
        st.write(f"DEBUG: Video ID extracted: {video_id}")
        return {"video_id":video_id}
    except Exception as e:
        st.error(f"Error in fetch_youtube_video_id: {str(e)}")
        raise

def fetch_youtube_transcript(graph:YoutubeStateGraph)->YoutubeStateGraph:
    try:
        st.write("DEBUG: Entering fetch_youtube_transcript")
        video_id = graph["video_id"]
        st.write(f"DEBUG: Fetching transcript for video_id: {video_id}")
        
        if not video_id or video_id.strip() == "":
            st.error("Invalid video ID extracted from URL")
            return {"transcript":""}
        
        try:
            # Check if YouTubeTranscriptApi is callable
            if YouTubeTranscriptApi is None:
                st.error("YouTubeTranscriptApi is None!")
                return {"transcript":""}
            
            st.write(f"DEBUG: YouTubeTranscriptApi type: {type(YouTubeTranscriptApi)}")
            
            # Try different methods based on youtube-transcript-api version
            try:
                # Method 1: list() method (newer versions)
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript_obj = transcript_list.find_transcript(['en'])
                transcript_data = transcript_obj.fetch()
            except:
                # Method 2: get_transcript() (some versions)
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                except:
                    # Method 3: Direct call (older versions)
                    transcript_data = YouTubeTranscriptApi().get_transcript(video_id, languages=['en'])
            
            transcript = ""
            for entry in transcript_data:
                transcript += entry["text"] + " "
            
            st.write(f"DEBUG: Transcript fetched, length: {len(transcript)}")
            return {"transcript":transcript}
        except TranscriptsDisabled:
            st.error("No captions available for this video.")
            return {"transcript":""}
        except Exception as e:
            st.error(f"Error fetching transcript: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return {"transcript":""}
    except Exception as e:
        st.error(f"Error in fetch_youtube_transcript: {str(e)}")
        raise

def spit_transcript(graph:YoutubeStateGraph)->YoutubeStateGraph:
    try:
        st.write("DEBUG: Entering spit_transcript")
        if graph["transcript"]=="":
            st.warning("Empty transcript, skipping chunking")
            return {"chunks":[]}
        
        transcript = graph["transcript"]
        st.write(f"DEBUG: Transcript length: {len(transcript)}")
        
        # Check if RecursiveCharacterTextSplitter is callable
        if RecursiveCharacterTextSplitter is None:
            st.error("RecursiveCharacterTextSplitter is None!")
            return {"chunks":[]}
        
        st.write(f"DEBUG: RecursiveCharacterTextSplitter type: {type(RecursiveCharacterTextSplitter)}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )
        chunks = splitter.split_text(transcript)
        st.write(f"DEBUG: Created {len(chunks)} chunks")
        return {"chunks":chunks}
    except Exception as e:
        st.error(f"Error in spit_transcript: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        raise

def create_embeddings(graph:YoutubeStateGraph)->YoutubeStateGraph:
    try:
        st.write("DEBUG: Entering create_embeddings")
        
        # Check if HuggingFaceEmbeddings is callable
        if HuggingFaceEmbeddings is None:
            st.error("HuggingFaceEmbeddings is None!")
            raise ValueError("HuggingFaceEmbeddings is not available")
        
        st.write(f"DEBUG: HuggingFaceEmbeddings type: {type(HuggingFaceEmbeddings)}")
        
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.write("DEBUG: Embeddings model created")
        
        chunks = graph["chunks"]
        st.write(f"DEBUG: Processing {len(chunks)} chunks")
        
        VECTORSTORE_PATH = "faiss_index"

        # Check if FAISS is callable
        if FAISS is None:
            st.error("FAISS is None!")
            raise ValueError("FAISS is not available")
        
        st.write(f"DEBUG: FAISS type: {type(FAISS)}")
        st.write("DEBUG: Generating new embeddings...")
        
        vector_store = FAISS.from_texts(chunks, embeddings_model)
        st.write("DEBUG: Vector store created")

        vector_store.save_local(VECTORSTORE_PATH)
        st.write("DEBUG: FAISS index saved locally")

        return {"vector_store": vector_store}
    except Exception as e:
        st.error(f"Error in create_embeddings: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        raise

def retriever(graph:YoutubeStateGraph)->YoutubeStateGraph:
    try:
        st.write("DEBUG: Entering retriever")
        retriever = graph["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k":4})
        st.write("DEBUG: Retriever created")
        return {"retriever":retriever}
    except Exception as e:
        st.error(f"Error in retriever: {str(e)}")
        raise

def augmentation(graph:YoutubeStateGraph)->YoutubeStateGraph:
    try:
        st.write("DEBUG: Entering augmentation")
        
        # Check if ChatGoogleGenerativeAI is callable
        if ChatGoogleGenerativeAI is None:
            st.error("ChatGoogleGenerativeAI is None!")
            raise ValueError("ChatGoogleGenerativeAI is not available")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.2
        )
        st.write("DEBUG: LLM created")
        
        context = graph["retriever"].invoke(graph["user_query"])
        st.write(f"DEBUG: Context retrieved, length: {len(context)}")
        
        template_string = """
        You are an AI assistant helping to answer user queries based on the context provided from a Youtube video transcript.
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Do not use your own knowledge or prior beliefs, only the context below.
        Context:
        {context}
        Question: {question}
        """
        
        if PromptTemplate is None:
            st.error("PromptTemplate is None!")
            raise ValueError("PromptTemplate is not available")
        
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
        st.write("DEBUG: Response generated")
        return {"llm_response":response}
    except Exception as e:
        st.error(f"Error in augmentation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        raise

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
                        st.write("DEBUG: Starting video load process")
                        
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
                        
                        st.write("DEBUG: Invoking workflow")
                        # Run workflow up to retriever setup
                        partial_state = app.invoke(inputs)
                        
                        st.write("DEBUG: Workflow completed")
                        # Store in session
                        st.session_state.current_url = youtube_url
                        st.session_state.workflow_state = partial_state
                        st.session_state.messages = []  # Clear chat history
                        
                        st.success("✅ Video loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading video: {str(e)}")
                        import traceback
                        st.error("Full traceback:")
                        st.code(traceback.format_exc())
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

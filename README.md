# YouTube Video Chat Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to chat with YouTube videos using natural language. The application extracts video transcripts, creates embeddings, and uses AI to answer questions based on the video content.

## Overview

This application uses LangChain, LangGraph, and Google's Gemini AI to create an intelligent chat interface for YouTube videos. It processes video transcripts, creates vector embeddings for efficient retrieval, and generates contextual responses to user queries.

## Features

- **YouTube Transcript Extraction**: Automatically fetches and processes video transcripts
- **Vector Embeddings**: Creates FAISS vector store for efficient semantic search
- **RAG Pipeline**: Implements Retrieval-Augmented Generation for accurate responses
- **Chat Interface**: ChatGPT-like conversational interface built with Streamlit
- **Session Management**: Maintains conversation history and video context
- **Dynamic Video Loading**: Load new videos and automatically create fresh embeddings
- **Context-Aware Responses**: Answers based strictly on video content

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Workflow**: LangGraph
- **Framework**: LangChain

## Architecture

The application follows a graph-based workflow with the following nodes:

1. **fetch_youtube_video_id**: Extracts video ID from YouTube URL
2. **fetch_youtube_transcript**: Retrieves video transcript using YouTube Transcript API
3. **split_transcript**: Splits transcript into chunks (500 characters with 200 overlap)
4. **create_embeddings**: Generates vector embeddings using HuggingFace model
5. **retriever**: Sets up similarity search retriever (k=4)
6. **augmentation**: Generates AI responses using retrieved context

## Installation

### Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini)
- HuggingFace API Token

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Kush-fanta/Youtube-Video-Chat-Assistant
cd youtube-video-chat-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your API keys there:
```bash
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token"
GOOGLE_API_KEY = "your_google_api_key"
```

## Requirements.txt

```
streamlit
langchain-core
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-huggingface
youtube-transcript-api
faiss-cpu
sentence-transformers
langgraph
python-dotenv
```

## Usage

### Running Locally

```bash
streamlit run app.py
```

### Using the Application

1. **Load a Video**:
   - Paste a YouTube URL in the sidebar
   - Click "Load Video" button
   - Wait for transcript extraction and embedding generation

2. **Ask Questions**:
   - Type your question in the chat input
   - Receive AI-generated responses based on video content
   - Continue the conversation with follow-up questions

3. **Load New Video**:
   - Paste a different YouTube URL
   - Previous embeddings are automatically deleted
   - New embeddings are created for the new video

4. **Clear History**:
   - Use "Clear Chat History" button to reset conversation



## Project Structure

```
youtube-video-chat-assistant/
├── app.py                     # Main application file
├── requirements.txt           # Python dependencies
├── .env                       # API keys (local only, not committed)
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── faiss_index/               # Vector store (created at runtime)
```

## Configuration

### Model Configuration

- **LLM Model**: Gemini 2.5 Flash
- **Temperature**: 0.2 (for consistent responses)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 200 characters
- **Retrieval**: Top 4 similar chunks

### Environment Variables

The application requires the following secrets:

- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API token for embeddings
- `GOOGLE_API_KEY`: Google API key for Gemini AI

## How It Works

### Transcript Processing

1. User provides YouTube URL
2. Application extracts video ID
3. Fetches English transcript using YouTube Transcript API
4. Splits transcript into manageable chunks

### Embedding Generation

1. Uses HuggingFace Sentence Transformers
2. Creates vector embeddings for each chunk
3. Stores embeddings in FAISS vector store
4. Saves to disk for persistence

### Query Processing

1. User asks a question
2. Question is converted to embedding
3. Retrieves 4 most similar transcript chunks
4. Sends chunks and question to Gemini AI
5. AI generates response based only on provided context
6. Response displayed in chat interface

## Limitations

- Only works with videos that have English captions/transcripts
- Requires active internet connection
- Limited by YouTube Transcript API availability
- Responses based solely on transcript content (no visual information)
- FAISS index stored locally (recreated on new deployment)

## Error Handling

The application includes error handling for:
- Missing video transcripts
- Invalid YouTube URLs
- API failures
- Embedding generation errors

## Security

- API keys stored in Streamlit secrets
- No API keys in source code
- `.gitignore` prevents secret files from being committed
- Environment variables properly sanitized

## Performance Considerations

- Initial video load may take 30-60 seconds
- Embedding generation depends on transcript length
- FAISS provides fast similarity search
- Subsequent queries on same video are faster

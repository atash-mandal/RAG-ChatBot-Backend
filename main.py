import os
import warnings
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import transformers

# Configuration
CACHE_DIR = "D:/hf_cache"
INDEX_PATH = "index/faiss_index"
MODEL_NAME = "google/flan-t5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model limits
MAX_INPUT_LENGTH = 400  # Safe limit for flan-t5-small (512 - buffer for prompt)
MAX_OUTPUT_LENGTH = 128

# Setup environment
os.environ["HF_HOME"] = CACHE_DIR
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables for models
vectorstore = None
qa_chain = None
tokenizer = None
llm_pipeline = None

def truncate_context(context: str, question: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Truncate context to fit within model's token limit"""
    global tokenizer
    
    if tokenizer is None:
        return context  # Fallback if tokenizer not available
    
    # Create the full prompt to test length
    prompt_template = f"""Based on the context provided, answer the question concisely.
If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:"""
    
    # Tokenize to check length
    tokens = tokenizer.encode(prompt_template, truncation=False)
    
    if len(tokens) <= max_length:
        return context
    
    # If too long, truncate context
    # Reserve tokens for the prompt structure and question
    reserved_tokens = 150  # Approximate tokens for prompt + question + answer
    available_tokens = max_length - reserved_tokens
    
    # Truncate context by splitting into sentences and keeping as many as possible
    sentences = context.split('. ')
    truncated_context = ""
    
    for sentence in sentences:
        test_context = truncated_context + sentence + ". "
        test_prompt = f"""Based on the context provided, answer the question concisely.
If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

Context: {test_context}

Question: {question}

Answer:"""
        
        test_tokens = tokenizer.encode(test_prompt, truncation=False)
        
        if len(test_tokens) <= max_length:
            truncated_context = test_context
        else:
            break
    
    return truncated_context.strip() or context[:500]  # Fallback to character limit

class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="The question to ask")

class Answer(BaseModel):
    answer: str
    sources_count: int

def load_models():
    """Load and initialize all models and components."""
    global vectorstore, qa_chain, tokenizer, llm_pipeline
    
    try:
        print("Loading embedding model...")
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        print("Loading FAISS index...")
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embedding, 
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3
        
        print("Loading language model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        
        llm_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_OUTPUT_LENGTH,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True,
            max_length=512  # Hard limit for the model
        )
        
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the context provided, answer the question concisely.
If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:"""
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        print("All models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up RAG API...")
    load_models()
    yield
    # Shutdown
    print("Shutting down RAG API...")

# FastAPI app
app = FastAPI(
    title="RAG Question Answering API",
    description="A retrieval-augmented generation API for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "RAG API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": qa_chain is not None,
        "cache_dir": CACHE_DIR,
        "transformers_cache": transformers.utils.default_cache_path
    }

@app.post("/query", response_model=Answer)
async def query_documents(data: Query) -> Answer:
    """Query the document collection."""
    if qa_chain is None or llm_pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        print(f"Processing query: {data.question[:50]}...")
        
        # Get relevant documents
        docs = vectorstore.similarity_search(data.question, k=5)
        context = " ".join([doc.page_content for doc in docs])
        
        # Truncate context to fit model limits
        truncated_context = truncate_context(context, data.question)
        
        # Create the input manually to ensure proper formatting
        input_text = f"""Based on the context provided, answer the question concisely.
If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

Context: {truncated_context}

Question: {data.question}

Answer:"""
        
        # Use the pipeline directly for better control
        result = llm_pipeline(input_text)
        answer = result[0]['generated_text'] if result else "I couldn't generate an answer."
        
        return Answer(
            answer=answer.strip(),
            sources_count=len(docs)
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        # Fallback to original qa_chain method
        try:
            result = qa_chain.invoke({"query": data.question})
            return Answer(
                answer=result["result"].strip(),
                sources_count=len(result.get("source_documents", []))
            )
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail="Error processing your question")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
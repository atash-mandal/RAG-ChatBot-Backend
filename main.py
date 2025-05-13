# import os
# os.environ["HF_HOME"] = "D:/hf_cache"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import transformers
import warnings

print("Cache dir:", transformers.utils.default_cache_path)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define input model
class Query(BaseModel):
    question: str

# Use local sentence transformer for embedding
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load local FAISS index
vectorstore = FAISS.load_local("index/faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Answer the question based ONLY on the context below.
    If you don't know the answer, respond with "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

# Load lightweight Seq2Seq model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="D:/hf_cache")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="D:/hf_cache")

# Setup text2text-generation pipeline
llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# QA chain using retriever + LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoint
@app.post("/query")
async def query(data: Query):
    print("data", data)
    result = qa_chain.invoke({"query": data.question})
    return {"answer": result["result"]}

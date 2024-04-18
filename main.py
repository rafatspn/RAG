import os
import torch
import weaviate
import warnings
import argparse
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

load_dotenv()

def create_db_instance():
    DB_URL = os.getenv("DB_URL")
    client = weaviate.Client(url=DB_URL)
    print("Connected to database")
    return client

def read_pdf(file_path):
    pdf_content = PyPDFLoader(file_path).load_and_split()
    return pdf_content

def chunk_pdf(pdf_content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = text_splitter.split_documents(pdf_content)
    return chunks

def get_emb_mode():
    emb_model = os.getenv("EMB_MODEL")
    model = HuggingFaceEmbeddings(model_name=emb_model)
    return model

def insert(client, chunks, emb_model):
    response = Weaviate.from_documents(documents=chunks, embedding=emb_model,client=client)
    print("Inserted Successfully")
    return response

def initialize_model(model_name):
    token = os.getenv("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        token=token,
    )
    return model

def initialize_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1 
    return tokenizer

def initialize_llm(model, tokenizer):
    llm = HuggingFacePipeline(pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        use_cache=True,
        max_length = 2048,
        do_sample=True,
        top_k=3,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    ))

    return llm

try:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Read a PDF file.")
    parser.add_argument("--pdf_file", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--query", type=str, required=True, help="Query to the model")
    args = parser.parse_args()

    pdf_file = args.pdf_file
    query = args.query

    pdf_contents = read_pdf(pdf_file)
    chunks = chunk_pdf(pdf_contents)
    emb_model = get_emb_mode()

    client = create_db_instance()
    storage = insert(client, chunks, emb_model)

    model_name = os.getenv("MODEL")
    model = initialize_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    llm = initialize_llm(model, tokenizer)

    retrieve_answer = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=storage.as_retriever()
    )

    resp = retrieve_answer.run(query)
    print(resp)
except Exception as e:
    print(e)

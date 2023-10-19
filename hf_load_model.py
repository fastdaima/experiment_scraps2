

model_path = "/home/srk/.cache/huggingface/hub/models--gpt2"

model_path= "gpt2"

import os 

from flask import Flask, request, jsonify, Response


import flask

from flask import * 

app = Flask(__name__)


import time 
from langchain.chains import LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.output_parsers import RegexParser

from transformers.generation import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import json
from langchain.llms import HuggingFacePipeline

def load_model():
    model_str = model_path 
    access_token = os.environ['HF_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained(model_str, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        device_map="auto",
        # quantization_config=bnb_config,
        local_files_only=True)

    streamer = TextStreamer(tokenizer)
    llm_pipeline = pipeline(
        "text-generation",  # task
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        do_sample=True,
        max_new_tokens=300,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        model_kwargs={"temperature": 0.01, "repetition_penalty": 2.5}
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm


print(load_model())
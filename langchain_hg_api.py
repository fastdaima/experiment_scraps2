

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


def query_llm(llm, query, task, prompt_template=None, vectordb=None):
    valid_tasks = ["instructive", "answer"]

    if task not in valid_tasks:
        raise ValueError(f"Invalid task '{task}'. Allowed values are {valid_tasks}")

    if task == "instructive":
        template = """
                    You are an intelligent chatbot. Answer the question posed by user.
                    Question: {question}
                    Answer:"""


    if task == "answer":
        if prompt_template:
            template = prompt_template
        else:
            template = """
                    You are an intelligent chatbot. Given the context below, answer the question given at the end:
                    Context: {context}
                    QUESTION: {question} 
                    Answer:"""

        PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        llm_chain = RetrievalQA.from_chain_type(llm,
                                                chain_type="stuff",
                                                # retriever=vectordb.as_retriever(),
                                                return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
        res = llm_chain({'query': query})
        # import pdb;pdb.set_trace()
        source_docs = [t.__dict__ for t in res["source_documents"]]

        return {"result": res["result"], "source_documents": source_docs}

    PROMPT = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(llm=llm, prompt=PROMPT, verbose=True)
    return {"result": llm_chain.run(question=query)}


@app.route('/llmgeneration_stream', methods=['POST'])
def llm_generation_stream():
    json_data = {"status": 0, "message": "Failed"}
    start = time.time()

    data = json.loads(flask.request.data)

    model = data.get("model", "gpt2")
    query = data.get("query")
    task = data.get("task", "instructive")
    prompt_template = data.get("prompt_template", None)

    valid_models = ["gpt2"]
    if model not in valid_models:
        raise ValueError(f"Invalid model '{model}'. Allowed values are {valid_models}")

    # try:
    if model == "gpt2":
        llm = load_model()
        print(llm)
    
    vectordb = None 
    return Response(stream_with_context(query_llm(llm, query, task, prompt_template, vectordb=vectordb)), mimetype='application/json')

    json_data["data"] = result
    json_data["message"] = "Passed"
    json_data["status"] = 1
    end = time.time()
    # logger.info("llm API: " + "Time: " + str(end - start) + "Model: " + model + "Query" + query + "text_generated"
    #             + str(result))

    print("llm API: " + "Time: " + str(end - start) + "Model: " + model + "Query" + query + "text_generated"
                + str(result))


    # except Exception as e:
    #     # logger.error(e, model, query)
    #     print(e, model, query)

    return flask.jsonify(json_data)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=False)
    

# curl -X POST -H "Content-Type: application/json" -d '{"query": "Once upon a time", "max_length": 100}' http://localhost:8095/llmgeneration_stream
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import FAISS

import json

from pathlib import Path

import unittest.mock as mock
import pdb

import boto3
import botocore
# import BedrockRuntime.Client.invoke_model(**kwargs)


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


client = boto3.client("bedrock-runtime")


HERE = Path(__file__).parent


def get_llm():

    model_kwargs =  { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": ["Human:"], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 },
     }

    return Bedrock(
        model_id="ai21.j2-ultra-v1",
        model_kwargs=model_kwargs,
    )


def get_index():
    """Return ..."""

    embeddings = BedrockEmbeddings()

    pdf_path = str(HERE / "borges-and-ai.pdf")
    # pdf_path = str(HERE / "manifesto_futurista.pdf")

    loader = PyPDFLoader(file_path=pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=text_splitter,
    )

    return index_creator.from_loaders([loader])


def get_memory():
    """Return ..."""

    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
    )



def MUTATE_invoke_model(client):
    def inspect_boto_calls(*args, **kwargs):
        body = kwargs["body"]
        body = json.loads(body)
        print_aside(body["prompt"])
        return client.__invoke_model(*args, **kwargs)
    client.__invoke_model = client.invoke_model
    client.invoke_model = inspect_boto_calls

TICKER = 0
COLORS = ["94", "96", "92"]

def print_aside(x):
    global TICKER
    color = COLORS[TICKER % len(COLORS)]
    TICKER += 1
    print(f"\n\033[{color}m{x}\033[0m\n")


def get_rag_chat_response(input_text, memory, index):

    llm = get_llm()

    MUTATE_invoke_model(llm.client)

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(
        llm,
        index.vectorstore.as_retriever(),
        memory=memory,
    )

    chat_response = conversation_with_retrieval.invoke({"question": input_text})

    return chat_response


def fuck_with_embedding():

    mem = get_memory()

    ind = get_index()

    try:
        while True:
            line = input("> ").strip()

            print("\033]0m", end="")

            if not line:
                continue

            res = get_rag_chat_response(line, mem, ind)

            print()
            print(res["answer"])
            print()

    except (KeyboardInterrupt, EOFError):
        print("Exiting...")
        raise SystemExit(0)

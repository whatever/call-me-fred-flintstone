import boto3
import json
import requests

from bs4 import BeautifulSoup


def fetch_text_from_url(url):
    """Return text from an HTML document"""

    resp = requests.get(url)

    resp = BeautifulSoup(resp.text, "html.parser")

    for script in resp(["script", "style"]):
        script.extract()

    text = resp.get_text()

    lines = (line.strip() for line in text.splitlines())

    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    text = "\n".join(chunk for chunk in chunks if chunk)

    return resp.get_text()


def query_llama(prompt):
    """Return generated text from bedrock"""

    client = boto3.client("bedrock-runtime")

    prompt = "\n".join([
        "[INST]Summarize this block of text:[/INST]",
        prompt,
    ])

    body = {
        "prompt": prompt,
        "temperature": 0.9,
        "top_p": 0.95,
        "max_gen_len": 2**8,
    }

    resp = client.invoke_model_with_response_stream(
        modelId="meta.llama2-13b-chat-v1",
        body=json.dumps(body),
    )

    result = ""

    for event in resp["body"]:
        chunk = event["chunk"]
        chunk = chunk["bytes"].decode("utf-8")
        chunk = json.loads(chunk)
        result += chunk["generation"]

    return result


def query_jurassic(prompt):
    """Return generated text from bedrock"""

    client = boto3.client("bedrock-runtime")

    prompt = "\n".join([
        "{text}Summarize this block of text with 5 bullet points:{/text}",
        prompt,
    ])

    model_kwargs = { #AI21
        "prompt": prompt,
        "maxTokens": 8000, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }

    resp = client.invoke_model(
        modelId="ai21.j2-ultra-v1",
        body=json.dumps(model_kwargs),
    )

    resp = resp["body"].read().decode("utf-8")
    resp = json.loads(resp)



    result = ""

    for chunk in resp["completions"]:
        result += chunk["data"]["text"]

    return result


def summarize_url(url):
    """Return a text summary of a URL"""
    text = fetch_text_from_url(url)
    return query_jurassic(text[:10_000]).strip()

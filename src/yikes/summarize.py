import boto3
import json
import logging
import requests

from bs4 import BeautifulSoup


def fetch(url):
    """Return the HTML from a URL"""

    resp = requests.get(url)
    return resp.text


def extract_text_from_html(html):
    """Return readable text from HTML"""

    resp = BeautifulSoup(html, "html.parser")

    for script in resp(["script", "style"]):
        script.extract()

    text = resp.get_text()

    lines = (
        line.strip()
        for line in text.splitlines()
    )

    chunks = (
        phrase.strip()
        for line in lines
        for phrase in line.split("  ")
    )

    return "\n".join(
        chunk
        for chunk in chunks
        if chunk
    )


def query_jurassic(prompt):
    """Return generated text from bedrock"""

    client = boto3.client("bedrock-runtime")

    prompt = "\n".join([
        # "{text}Summarize this block of text with 5 bullet points:{/text}",
        # prompt,
        "The following is text from a website:",
        prompt,
        "Summarize the website with 5 bullet points:",
    ])

    model_kwargs = { #AI21
        "prompt": prompt,
        "maxTokens": 3000, 
        "temperature": 0.3, 
        "topP": 0.7, 
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


from collections import namedtuple

Summarization = namedtuple(
    "Summarization",
    ["url", "title", "image", "summary"],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def summarize_url(url):
    """Return a text summary of a URL"""

    logger.info("fetching %url", url)
    html = fetch(url)

    print("extracting text from HTML")
    text = extract_text_from_html(html)

    print("text is:", text)

    print("summarizing text with bedrock")
    summary = query_jurassic(text[:10_000]).strip()

    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("meta", property="og:title")
    image = soup.find("meta", property="og:image")
    description = soup.find("meta", property="og:description")

    return {
        "url": url,
        "title": title["content"] if title else "n/a",
        "image": image["content"] if image else None,
        "description": description["content"] if description else "n/a",
        "summary": summary,
    }


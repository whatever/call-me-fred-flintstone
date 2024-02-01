import argparse
import boto3
import json
import os


from yikes.summarize import (
    summarize_url,
    fetch_text_from_url,
)


from aiohttp import web


HERE = os.path.realpath(os.path.dirname(__file__))


def query_llama(prompt):

    client = boto3.client("bedrock-runtime")

    prompt = "\n".join([
        "[INST]You are a coder[/INST]",
        prompt,
    ])

    body = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_gen_len": 2**8,
    }

    result = ""

    resp = client.invoke_model_with_response_stream(
        modelId="meta.llama2-13b-chat-v1",
        body=json.dumps(body),
    )

    for event in resp["body"]:
        chunk = event["chunk"]
        chunk = chunk["bytes"].decode("utf-8")
        chunk = json.loads(chunk)
        result += chunk["generation"]

    return result


def webapp():

    async def root_handler(request):
        return web.FileResponse(f"{HERE}/static/index.html")

    async def query_preview_handler(request):
        return web.Respose(text="Preview")

    async def query_handler(request):
        params = await request.json()
        query = params["query"]
        return web.Response(text=query_llama(query))

    app = web.Application()

    app.add_routes([
        web.get("/", root_handler),
        web.post("/query", query_handler),
        web.get("/preview", query_preview_handler),
        web.static("/", f"{HERE}/static"),
    ])

    return app




def main():
    """Serve website to query LLaMa"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8181)
    # parser.add_argument("--temperature", type=float, default=0.5)
    # parser.add_argument("--top-p", type=float, default=0.9)
    # parser.add_argument("--max-gen-len", type=int, default=2**8)
    args = parser.parse_args()

    url = "https://openai.com/research/building-an-early-warning-system-for-llm-aided-biological-threat-creation"

    print(summarize_url(url))

    return


    app = webapp()

    web.run_app(
        app,
        print=None,
        port=args.port,
    )

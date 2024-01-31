import argparse
import boto3
import json
import os


from aiohttp import web


HERE = os.path.realpath(os.path.dirname(__file__))

def webapp():

    async def root_handler(request):
        return web.FileResponse(f"{HERE}/static/index.html")

    async def query_preview_handler(request):
        return web.Respose(text="Preview")

    async def query_handler(request):
        return web.Response(text="query")

    app = web.Application()

    app.add_routes([
        web.get("/", root_handler),
        web.get("/query", query_handler),
        web.get("/preview", query_preview_handler),
        web.static("/", f"{HERE}/static"),
    ])

    return app




def main():
    """You know..."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-gen-len", type=int, default=2**8)
    parser.add_argument("prompt", type=str)
    args = parser.parse_args()


    app = webapp()

    web.run_app(
        app,
        print=None,
        port=8181,
    )

    return

    client = boto3.client("bedrock-runtime")

    prompt = "\n".join([
        "[INST]You are a coder[/INST]",
        args.prompt,
    ])

    body = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_gen_len": 2**8,
    }

    resp = client.invoke_model_with_response_stream(
        modelId="meta.llama2-13b-chat-v1",
        body=json.dumps(body),
    )

    for event in resp["body"]:
        chunk = event["chunk"]
        chunk = chunk["bytes"].decode("utf-8")
        chunk = json.loads(chunk)
        print(chunk["generation"], end="")

import os
import json
import time
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)


def get_dynamic_knowledge(prompt: str, api_key: str | None = None) -> str:
    """Fetch dynamic knowledge using OpenAI as fluid weights."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message["content"]


@app.post("/generate")
def generate() -> tuple[dict[str, str], int]:
    payload = request.get_json(force=True)
    prompt = payload.get("prompt", "")
    response_text = get_dynamic_knowledge(prompt)
    log_entry = {"prompt": prompt, "response": response_text, "timestamp": time.time()}
    os.makedirs("logs/server", exist_ok=True)
    with open("logs/server/latest.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    return jsonify({"response": response_text}), 200


if __name__ == "__main__":  # pragma: no cover - dev server
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

import requests
from pathlib import Path
from typing import List, Dict, Optional

API_KEY = Path("/Users/bo/workspace/Deep_Research/api.key").read_text().strip()


def call_siliconflow_llm(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "deepseek-ai/DeepSeek-V3.2",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    stream: bool = False,
    timeout: int = 60,
) -> str:
    """
    Call SiliconFlow Chat Completions API.

    Parameters
    ----------
    api_key : str
        Your SiliconFlow API key
    messages : List[Dict[str, str]]
        Chat messages, e.g.:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain transformers briefly."}
        ]
    model : str
        Model name, default: deepseek-ai/DeepSeek-V3.2
    temperature : float
        Sampling temperature
    max_tokens : int
        Maximum number of tokens to generate
    top_p : float
        Nucleus sampling parameter
    stream : bool
        Whether to use streaming responses
    timeout : int
        HTTP timeout in seconds

    Returns
    -------
    str
        Assistant response text
    """

    url = "https://api.siliconflow.cn/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()

    # Non-streaming response
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":

    messages = [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is DeepSeek-V3.2 good at?"},
    ]

    reply = call_siliconflow_llm(
        api_key=API_KEY,
        messages=messages,
        temperature=0.3,
    )

    print(reply)

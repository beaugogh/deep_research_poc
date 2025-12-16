import os
from openai import OpenAI
import logging
import requests
import yaml
import httpx
from urllib.request import getproxies
from collections.abc import Generator
from typing import List, Dict, Optional


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# add module logger
logger = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_CLIENT = None
CONFIG = None


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        logger.info(f"Error: File '{file_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        logger.info(f"Error parsing YAML file: {e}")
        return {}


def bypass_proxies():
    proxies = getproxies()
    http_proxy = proxies.get("http") or proxies.get("https")
    if http_proxy:
        # export proxy only if a proxy is found
        os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = http_proxy
        os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = http_proxy

    # Ensure localhost/loopback bypass the proxy
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    for host in (".huawei.com", "localhost", "127.0.0.1", "::1"):
        if host not in no_proxy:
            no_proxy = (no_proxy + "," + host) if no_proxy else host
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = no_proxy


def initialize():
    global LLM_CLIENT
    global CONFIG
    bypass_proxies()
    CONFIG = read_yaml(os.path.join(CUR_DIR, "config.yaml"))
    llm_config = CONFIG.get("llm", {})
    token = llm_config.get("api_key", "")
    url = llm_config.get("url", "")
    headers = {"X-Auth-Token": token}
    LLM_CLIENT = OpenAI(
        base_url=url,
        api_key=str(token),
        default_headers=headers,
        http_client=httpx.Client(verify=False),
    )


def _get_chat_entry(prompt, role="user"):
    return {"role": role, "content": prompt}


def _format_messages(model_id, prompt=None, messages=None, system_prompt=""):
    if prompt:
        if "qwen" in model_id.lower():
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                if system_prompt
                else f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )
            _messages = [_get_chat_entry(prompt=prompt)]
        else:
            _messages = (
                [
                    _get_chat_entry(prompt=system_prompt, role="system"),
                    _get_chat_entry(prompt=prompt),
                ]
                if system_prompt
                else [_get_chat_entry(prompt=prompt)]
            )
    elif messages and isinstance(messages, list) and len(messages) > 0:
        if "qwen" in model_id.lower():
            _prompt = ""
            for msg in messages:
                _prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            _prompt += "<|im_start|>assistant\n"
            _messages = [_get_chat_entry(prompt=prompt)]
        else:
            _messages = messages

    return _messages


def _call_llm_api(prompt=None, messages=None, system_prompt="", stream=False):
    if LLM_CLIENT is None:
        raise RuntimeError("you need to initiate CLIENT")

    config = CONFIG.get("llm", {})
    model_id = config["model_name"]
    t = config["temperature"]
    top_p = config["top_p"]
    _messages = _format_messages(
        model_id=model_id, prompt=prompt, messages=messages, system_prompt=system_prompt
    )
    logger.info(f"\ncalling {model_id}, with t={t}")
    stream_itr = LLM_CLIENT.chat.completions.create(
        model=model_id,
        messages=_messages,
        temperature=t,
        top_p=top_p,
        stream=stream,
        user="",
    )
    return stream_itr


def call_llm_stream(prompt=None, messages=None, system_prompt="") -> Generator:
    stream_itr = _call_llm_api(
        prompt=prompt, messages=messages, system_prompt=system_prompt, stream=True
    )
    for chunk in stream_itr:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content


def call_llm(prompt=None, messages=None, system_prompt=""):
    stream_itr = _call_llm_api(
        prompt=prompt, messages=messages, system_prompt=system_prompt, stream=False
    )
    result = stream_itr.choices[0].message.content
    return result


def _resolve_verify_flag() -> Optional[str | bool]:
    """Determine TLS verification behavior.

    Priority:
    1) REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE (path to CA bundle)
    2) SILICONFLOW_VERIFY=false|0 to disable verification
    3) Default: False (to accommodate corporate MITM proxies)
    """
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if bundle:
        return bundle
    verify_env = os.environ.get("SILICONFLOW_VERIFY")
    if verify_env is not None and verify_env.lower() in {"0", "false", "no"}:
        return False
    return False


def call_siliconflow_llm(
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

    url = CONFIG.get("silicon_flow", {}).get("url", "")
    api_key = CONFIG.get("silicon_flow", {}).get("api_key", "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resolved_model = os.environ.get("SILICONFLOW_MODEL", model)

    payload = {
        "model": resolved_model,
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
        verify=_resolve_verify_flag(),
    )

    if response.status_code >= 400:
        # Surface server-provided error details to aid debugging.
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise requests.HTTPError(
            f"SiliconFlow API error {response.status_code}: {detail}",
            response=response,
        )

    data = response.json()

    # Non-streaming response
    return data["choices"][0]["message"]["content"]


def test_silicon_flow_llm():
    initialize()
    messages = [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is DeepSeek-V3.2 good at?"},
    ]

    reply = call_siliconflow_llm(messages=messages)
    print(reply)


def test_llm():
    initialize()
    test_prompt = "hello, who are you?"
    full_res = ""
    for chunk in call_llm_stream(prompt=test_prompt):
        full_res += chunk
        print(chunk, end="", flush=True)

    print("\n\nHere is the full response: ")
    print(full_res)


if __name__ == "__main__":

    print()

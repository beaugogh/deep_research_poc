import os
from openai import OpenAI, AsyncOpenAI
import logging
import httpx
from urllib.request import getproxies
from collections.abc import Generator
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from utils import read_yaml


logger = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OPENAI_CLIENT = None
OPENAI_ASYNC_CLIENT = None
CONFIG = None


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
    global OPENAI_CLIENT
    global OPENAI_ASYNC_CLIENT
    global CONFIG
    bypass_proxies()
    CONFIG = read_yaml(os.path.join(CUR_DIR, "config.yaml"))
    llm_config = CONFIG.get("llm", {})
    token = llm_config.get("api_key", "")
    url = llm_config.get("url", "")
    headers = {
        "X-Auth-Token": token,
        "X-HW-ID": llm_config.get("x_hw_id", ""),
        "X-HW-AppKey": llm_config.get("x_hw_appkey", ""),
    }
    OPENAI_CLIENT = OpenAI(
        base_url=url,
        api_key=str(token),
        default_headers=headers,
        http_client=httpx.Client(verify=False),
    )
    OPENAI_ASYNC_CLIENT = AsyncOpenAI(
        base_url=url,
        api_key=str(token),
        default_headers=headers,
        http_client=httpx.AsyncClient(verify=False),
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
        # TODO: check if messages is in the right format
        _messages = messages

    return _messages


def _call_llm_api(prompt=None, messages=None, system_prompt="", stream=False):
    if OPENAI_CLIENT is None:
        # raise RuntimeError("you need to initiate CLIENT")
        initialize()

    config = CONFIG.get("llm", {})
    model_id = config["model_name"]
    t = config["temperature"]
    top_p = config["top_p"]
    _messages = _format_messages(
        model_id=model_id, prompt=prompt, messages=messages, system_prompt=system_prompt
    )
    logger.info(f"\ncalling {model_id}, with t={t}")
    stream_itr = OPENAI_CLIENT.chat.completions.create(
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


def call_tavily_api(query: str, max_results: int = 3, search_depth: str = "basic"):
    url = CONFIG.get("tavily", {}).get("url", "https://api.tavily.com/search")
    api_key = CONFIG.get("tavily", {}).get("api_key", "")
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }
    response = requests.post(url, json=payload, timeout=30, verify=False)
    response.raise_for_status()
    results = response.json().get("results", [])
    return results


def test_llm():
    initialize()
    test_prompt = "hello, who are you?"
    full_res = ""
    for chunk in call_llm_stream(prompt=test_prompt):
        full_res += chunk
        print(chunk, end="", flush=True)

    logger.info("\n\nHere is the full response: ")
    logger.info(full_res)


def test_tavily_search():
    initialize()
    results = call_tavily_api("latest advancements in LLM agents")
    for item in results:
        print(
            f"""--------------------------
Title: {item["title"]}
URL: {item["url"]}
CONTENT: {item["content"]}
SCORE: {item["score"]}
--------------------------"""
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger.info("start")
    # test_llm()
    test_tavily_search()
    logger.info("finish")

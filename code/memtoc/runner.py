"""Model clients. v0: OpenAI-compatible HTTP (against vLLM on an A100) + DryRun.

DryRun exists to validate the pipe without a model: it answers with the tool
output when there is one, otherwise with a marker. Real runs happen only on
the server (long runs go to the background with checkpoints).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request


class DryRunClient:
    """Echo client: the final answer is the tool output or a stub.
    
    If the prompt is extraction-first (it asks for 'FINAL:'), the answer is
    wrapped in a reasoning block plus a FINAL: line — so the v1 pipe can be
    run without a model. v0 prompts contain no 'FINAL:', so v0 behaviour does
    not change.
    """

    name = "dryrun"

    def complete(self, prompt: str) -> str:
        marker = '"result": '
        ans = "[dryrun-no-tool]"
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            try:
                ans = json.loads("{" + marker + tail.split("}", 1)[0] + "}")["result"]
            except Exception:
                pass
        if re.search(r"final\s*:", prompt, re.IGNORECASE):
            return f"(dryrun reasoning)\nFINAL: {ans}"
        return ans


class OpenAICompatClient:
    """Minimal chat.completions client (vLLM and compatible servers)."""

    def __init__(self, model: str, base_url: str | None = None, timeout: int = 120,
                 max_tokens: int = 128, seed: int | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get(
            "KCB_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = os.environ.get("KCB_API_KEY", "EMPTY")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.seed = seed
        self.name = model

    def complete(self, prompt: str) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                **({"seed": self.seed} if self.seed is not None else {}),
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.load(r)
        return body["choices"][0]["message"]["content"]


class CompletionsClient:
    """A /v1/completions client for base models without a chat template.
    
    Episode prompts end in 'Answer:' — a base model continues the text.
    stop='\n': the answer is one line; without the stop a base model runs on
    into self-continuation (the next 'Question:'), which under substring
    matching produces false hits.
    """

    def __init__(self, model: str, base_url: str | None = None, timeout: int = 120,
                 max_tokens: int = 128, stop: list[str] | None = None,
                 seed: int | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get(
            "KCB_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = os.environ.get("KCB_API_KEY", "EMPTY")
        self.timeout = timeout
        self.max_tokens = max_tokens
        # v0: stop=['\n'] (one line). v1 extraction-first: stop=None (the
        # answer is
        # multi-line and ends in FINAL:) — pass it explicitly.
        self.stop = ["\n"] if stop is None else stop
        self.seed = seed
        self.name = f"completion:{model}"

    def complete(self, prompt: str) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/completions",
            data=json.dumps({
                "model": self.model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                "stop": self.stop,
                **({"seed": self.seed} if self.seed is not None else {}),
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.load(r)
        return body["choices"][0]["text"].strip()


def get_client(spec: str, max_tokens: int = 128,
               completion_stop: list[str] | None = None,
               seed: int | None = None):
    """spec: 'dryrun', 'completion:<model>' (base) or a model name (chat).
    
    max_tokens/completion_stop are for v1 (extraction-first: longer, no
    stop=\n).
    """
    if spec == "dryrun":
        return DryRunClient()
    if spec.startswith("completion:"):
        return CompletionsClient(spec.removeprefix("completion:"),
                                 max_tokens=max_tokens, stop=completion_stop,
                                 seed=seed)
    return OpenAICompatClient(spec, max_tokens=max_tokens, seed=seed)

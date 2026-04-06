import base64
import json
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request


def require_api_key(api_key: str = "", api_key_env: str = "") -> str:
    if api_key:
        return api_key
    if api_key_env:
        value = os.environ.get(api_key_env, "").strip()
        if value:
            return value
        raise RuntimeError(f"Missing API key in environment variable: {api_key_env}")
    raise RuntimeError("Missing API key configuration.")


def encode_image_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def encode_image_inline(image_path: str) -> Dict[str, str]:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"mime_type": mime_type, "data": data}


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 60,
    max_retries: int = 6,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(max_retries):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries - 1:
                retry_after_header = exc.headers.get("Retry-After", "").strip()
                retry_after = None
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = None
                if retry_after is None:
                    match = re.search(r"try again in ([\d.]+)s", details, flags=re.IGNORECASE)
                    if match:
                        retry_after = float(match.group(1))
                if retry_after is None:
                    retry_after = min(8.0, 1.0 * (2 ** attempt))
                retry_after += random.uniform(0.05, 0.25)
                print(f"[WARN] HTTP 429 calling {url}; retrying in {retry_after:.2f}s ({attempt + 1}/{max_retries})")
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {details}") from exc
        except error.URLError as exc:
            if attempt < max_retries - 1:
                wait = min(8.0, 1.0 * (2 ** attempt)) + random.uniform(0.05, 0.25)
                print(f"[WARN] Network error calling {url}: {exc}; retrying in {wait:.2f}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc
    raise RuntimeError(f"HTTP request failed after {max_retries} retries calling {url}")

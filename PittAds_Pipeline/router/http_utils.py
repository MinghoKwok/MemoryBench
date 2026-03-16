import base64
import json
import mimetypes
import os
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
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc

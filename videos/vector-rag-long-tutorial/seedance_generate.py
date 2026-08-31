#!/usr/bin/env python3
"""预览或提交 Seedance 开场任务，并轮询下载生成视频。"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = Path(__file__).with_name("seedance-opening.txt")


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator:
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def request_json(
    method: str, url: str, api_key: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Seedance API 返回 HTTP {error.code}：{detail}") from error
    except URLError as error:
        raise RuntimeError(f"无法连接 Seedance API：{error.reason}") from error
    if not isinstance(result, dict):
        raise RuntimeError("Seedance API 返回格式不是 JSON 对象")
    return result


def first_string(value: Any, keys: tuple[str, ...]) -> str:
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        for child in value.values():
            found = first_string(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_string(child, keys)
            if found:
                return found
    return ""


def video_url(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("video_url", "url"):
            item = value.get(key)
            if isinstance(item, str) and item.startswith(("https://", "http://")):
                return item
        for child in value.values():
            found = video_url(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = video_url(child)
            if found:
                return found
    return ""


def download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "lob-vector-seedance/1.0"})
    with urlopen(request, timeout=180) as response:
        target.write_bytes(response.read())


def main() -> int:
    load_env(ROOT / ".env")
    base_url = os.getenv("SEEDANCE_API_BASE", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
    output_dir = ROOT / os.getenv(
        "SEEDANCE_OUTPUT_DIR", "videos/vector-rag-long-tutorial/seedance"
    )
    payload = {
        "model": os.getenv("SEEDANCE_MODEL", "doubao-seedance-2-5-260628"),
        "content": [{"type": "text", "text": PROMPT_PATH.read_text(encoding="utf-8").strip()}],
        "duration": int(os.getenv("SEEDANCE_DURATION", "10")),
        "ratio": os.getenv("SEEDANCE_RATIO", "9:16"),
        "resolution": os.getenv("SEEDANCE_RESOLUTION", "720p"),
        "generate_audio": env_bool("SEEDANCE_AUDIO", True),
        "watermark": False,
    }
    preview = {
        "submit": env_bool("SEEDANCE_SUBMIT"),
        "endpoint": f"{base_url}/contents/generations/tasks",
        "api_key": "<已配置>" if os.getenv("SEEDANCE_ARK_API_KEY") else "<未配置>",
        "payload": payload,
        "output": str(output_dir / "opening.mp4"),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not preview["submit"]:
        print("\n预览完成：SEEDANCE_SUBMIT=0，未提交付费任务。")
        return 0

    api_key = os.getenv("SEEDANCE_ARK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SEEDANCE_SUBMIT=1，但 SEEDANCE_ARK_API_KEY 未配置")
    output_dir.mkdir(parents=True, exist_ok=True)
    created = request_json(
        "POST", f"{base_url}/contents/generations/tasks", api_key, payload
    )
    task_id = first_string(created, ("id", "task_id"))
    if not task_id:
        raise RuntimeError("Seedance 创建响应缺少任务 ID")
    (output_dir / "create-response.json").write_text(
        json.dumps(created, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"task_id={task_id}")

    timeout = int(os.getenv("SEEDANCE_TIMEOUT_SECONDS", "1200"))
    interval = int(os.getenv("SEEDANCE_POLL_SECONDS", "10"))
    deadline = time.monotonic() + timeout
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        final = request_json(
            "GET", f"{base_url}/contents/generations/tasks/{task_id}", api_key
        )
        status = first_string(final, ("status",)).lower() or "unknown"
        print(f"status={status}")
        if status in {"succeeded", "completed"}:
            break
        if status in {"failed", "cancelled", "canceled", "expired"}:
            message = first_string(final, ("message", "error_message"))
            raise RuntimeError(f"Seedance 任务失败：{status} {message}".strip())
        time.sleep(interval)
    else:
        raise RuntimeError(f"等待 Seedance 任务超时：{timeout} 秒")

    (output_dir / "final-response.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    url = video_url(final)
    if not url:
        raise RuntimeError("Seedance 成功响应中没有找到视频地址")
    target = output_dir / "opening.mp4"
    download(url, target)
    print(f"已下载：{target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(1)

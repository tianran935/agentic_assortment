from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import requests


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.4-image-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-LLM shelf image generation and editing via OpenRouter Image 2."
    )
    parser.add_argument("--request-file", type=Path, required=True, help="JSON file describing the shelf request.")
    parser.add_argument("--output-file", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter image model.")
    parser.add_argument("--aspect-ratio", default="4:3", help="Image aspect ratio.")
    parser.add_argument("--image-size", default="1K", choices=["1K", "2K"], help="Image size.")
    parser.add_argument("--timeout-seconds", type=int, default=360, help="HTTP timeout in seconds.")
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_local_image(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def format_sku_lines(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        promo = item.get("promotion", "none")
        price = item.get("price", "unknown")
        lines.append(
            f'- {item["sku_id"]}: item="{item["item"]}", price="{price}", promotion="{promo}", '
            f'position=(row {item["position"]["row"]}, col {item["position"]["col"]})'
        )
    return "\n".join(lines)


def build_generate_prompt(payload: dict[str, Any]) -> str:
    style = payload.get("style", "clean grocery shelf experiment image")
    category = payload.get("category", "grocery")
    notes = payload.get(
        "notes",
        "The shelf should be front-facing, visually clean, and easy to read. Distinguish products clearly.",
    )
    return (
        f"Generate one {style} for category {category}. "
        f"{notes} Respect the following structured shelf configuration exactly as much as possible.\n"
        f"{format_sku_lines(payload['skus'])}\n"
        "Render a realistic grocery shelf photograph that matches a real supermarket shelf. "
        "The shelf should be densely stocked and visually full, with products filling almost all visible facing space. "
        "Avoid large empty gaps or obviously sparse experimental layouts unless a gap is explicitly requested. "
        "Use repeated facings and neighboring filler products from the same category when needed so the shelf looks naturally merchandised. "
        "Keep the requested target SKUs at their specified positions and preserve their item identity, price cue, and promotion type. "
        "Make price tags and promotion markers visible and believable. "
        "The final image should look like a real fully merchandised cereal shelf in a supermarket rather than a minimal mockup."
    )


def build_edit_prompt(payload: dict[str, Any]) -> str:
    notes = payload.get(
        "notes",
        "Edit the provided shelf image while preserving the overall shelf framing and realism as much as possible.",
    )
    return (
        f"Edit the provided grocery shelf image. {notes} "
        "Update it to reflect the following structured shelf configuration exactly as much as possible.\n"
        f"{format_sku_lines(payload['skus'])}\n"
        "Keep it as a coherent shelf photograph and change only what is needed to match the new item, price, promotion, and position instructions."
    )


def build_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    mode = payload["mode"]
    if mode == "generate":
        return [{"role": "user", "content": build_generate_prompt(payload)}]

    if mode == "edit":
        input_image_path = Path(payload["input_image"])
        if not input_image_path.exists():
            raise FileNotFoundError(f"Input image not found for edit mode: {input_image_path}")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_edit_prompt(payload)},
                    {"type": "image_url", "image_url": {"url": encode_local_image(input_image_path)}},
                ],
            }
        ]

    raise ValueError(f"Unsupported mode: {mode}")


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    aspect_ratio: str,
    image_size: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.codex.workflow",
            "X-OpenRouter-Title": "Shelf Image LLM Workflow",
        },
        json={
            "model": model,
            "messages": messages,
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            },
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter image request failed with status {response.status_code}: {response.text}")
    return response.json()


def extract_image_bytes(result: dict[str, Any]) -> bytes:
    message = result["choices"][0]["message"]
    images = message.get("images") or []
    if not images:
        raise ValueError("No generated image found in OpenRouter response.")
    image_url = images[0]["image_url"]["url"]
    if not image_url.startswith("data:image"):
        raise ValueError("Expected a base64 image data URL in the response.")
    return base64.b64decode(image_url.split(",", 1)[1])


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    payload = load_request(args.request_file)
    messages = build_messages(payload)
    result = call_openrouter(
        api_key=api_key,
        model=args.model,
        messages=messages,
        aspect_ratio=args.aspect_ratio,
        image_size=args.image_size,
        timeout_seconds=args.timeout_seconds,
    )
    image_bytes = extract_image_bytes(result)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_bytes(image_bytes)

    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "model": args.model,
                "output_file": str(args.output_file),
                "request_file": str(args.request_file),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

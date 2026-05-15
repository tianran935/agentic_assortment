from __future__ import annotations

import argparse
import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TEXT_MODEL = "qwen/qwen-2.5-72b-instruct"
DEFAULT_IMAGE_MODEL = "qwen/qwen2.5-vl-72b-instruct"


TEXT_INSTRUCTION = """You are evaluating a grocery shopping choice task.

Pick exactly one product from the candidate list.
Return JSON only with this schema:
{
  "selected_product_id": "string",
  "selected_product_name": "string",
  "reasoning_summary": "brief explanation",
  "confidence": "low|medium|high"
}

Decision rule:
- Choose one product that best fits the shopping request.
- Use only the products provided in the candidate list.
- Do not include markdown fences or extra commentary.
"""


IMAGE_INSTRUCTION = """You are evaluating a grocery shelf image.

Pick exactly one product from the candidate list based on the shelf image and the shopping request.
Return JSON only with this schema:
{
  "selected_product_id": "string",
  "selected_product_name": "string",
  "reasoning_summary": "brief explanation",
  "confidence": "low|medium|high"
}

Decision rule:
- Use the image as primary evidence.
- Use only the candidate products provided in the list.
- Do not include markdown fences or extra commentary.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Part A + Part B LLM product choice workflow.")
    parser.add_argument("--mode", choices=["text", "image", "both"], default="both", help="Which part to run.")
    parser.add_argument(
        "--products-file",
        type=Path,
        default=Path(__file__).with_name("demo_products.json"),
        help="JSON file containing candidate products.",
    )
    parser.add_argument(
        "--image-file",
        type=Path,
        default=Path(__file__).with_name("demo_shelf.png"),
        help="Shelf image for Part B.",
    )
    parser.add_argument(
        "--shopping-request",
        default="Choose one beverage for a shopper who wants a healthy everyday drink with low sugar preference.",
        help="User shopping request shared by Part A and Part B.",
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL, help="OpenRouter model id for Part A.")
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL, help="OpenRouter model id for Part B.")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path(__file__).with_name(".env.openrouter"),
        help="Optional file that stores OPENROUTER_API_KEY=... .",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).with_name("demo_run_output.json"),
        help="Where to write the final combined JSON output.",
    )
    return parser.parse_args()


def load_api_key(api_key_file: Path) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key
    if api_key_file.exists():
        for line in api_key_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY is not set and no usable api key file was found.")


def load_products(path: Path) -> list[dict[str, Any]]:
    products = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(products, list) or not products:
        raise ValueError("products file must contain a non-empty JSON list")
    return products


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type = "image/png"
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    raw = image_path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def parse_json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_message_text(payload: dict[str, Any]) -> str:
    message = payload["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_parts = []
        for item in message:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts).strip()
    raise ValueError(f"Unexpected message content type: {type(message)!r}")


def call_openrouter(api_key: str, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.codex.workflow",
            "X-OpenRouter-Title": "LLM Choice Workflow Demo",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            }
        ),
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter request failed with status {response.status_code}: {response.text}")
    return response.json()


def validate_selection(selection: dict[str, Any], products: list[dict[str, Any]]) -> None:
    valid_ids = {product["id"] for product in products}
    if selection.get("selected_product_id") not in valid_ids:
        raise ValueError(f"Model selected an unknown product id: {selection.get('selected_product_id')}")


def normalize_result(
    *,
    part: str,
    input_type: str,
    selection: dict[str, Any],
    products: list[dict[str, Any]],
    shopping_request: str,
) -> dict[str, Any]:
    selected_id = selection["selected_product_id"]
    selected_product = next(product for product in products if product["id"] == selected_id)
    return {
        "part": part,
        "input_type": input_type,
        "shopping_request": shopping_request,
        "selected_product": {
            "id": selected_product["id"],
            "name": selected_product["name"],
            "brand": selected_product["brand"],
            "category": selected_product["category"],
            "price": selected_product["price"],
            "size": selected_product["size"],
        },
        "reasoning_summary": selection["reasoning_summary"],
        "confidence": selection["confidence"],
    }


def run_part_a(api_key: str, model: str, products: list[dict[str, Any]], shopping_request: str) -> dict[str, Any]:
    payload = call_openrouter(
        api_key,
        model,
        [
            {"role": "system", "content": TEXT_INSTRUCTION},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "shopping_request": shopping_request,
                        "candidate_products": products,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
    )
    selection = parse_json_from_text(extract_message_text(payload))
    validate_selection(selection, products)
    return normalize_result(
        part="A",
        input_type="product_list_text",
        selection=selection,
        products=products,
        shopping_request=shopping_request,
    )


def run_part_b(
    api_key: str,
    model: str,
    products: list[dict[str, Any]],
    image_path: Path,
    shopping_request: str,
) -> dict[str, Any]:
    image_url = encode_image_as_data_url(image_path)
    payload = call_openrouter(
        api_key,
        model,
        [
            {"role": "system", "content": IMAGE_INSTRUCTION},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "shopping_request": shopping_request,
                                "candidate_products": products,
                                "note": "Choose one item visible on the shelf image.",
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
    )
    selection = parse_json_from_text(extract_message_text(payload))
    validate_selection(selection, products)
    return normalize_result(
        part="B",
        input_type="shelf_image",
        selection=selection,
        products=products,
        shopping_request=shopping_request,
    )


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.api_key_file)
    products = load_products(args.products_file)
    results: dict[str, Any] = {}

    if args.mode in {"text", "both"}:
        results["part_a"] = run_part_a(api_key, args.text_model, products, args.shopping_request)
    if args.mode in {"image", "both"}:
        if not args.image_file.exists():
            raise FileNotFoundError(f"Image file not found: {args.image_file}")
        results["part_b"] = run_part_b(api_key, args.image_model, products, args.image_file, args.shopping_request)

    output = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "part_a": args.text_model if args.mode in {"text", "both"} else None,
            "part_b": args.image_model if args.mode in {"image", "both"} else None,
        },
        "workflow_status": "success",
        "results": results,
    }
    args.output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

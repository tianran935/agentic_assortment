from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CARD_COLORS = [
    "#D94841",
    "#F59E0B",
    "#2E86AB",
    "#4CAF50",
    "#7B61FF",
    "#E76F51",
]


def load_products(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def pick_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        if draw.textlength(trial, font=font) <= width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def build_demo_shelf(products: list[dict], output_path: Path) -> None:
    image = Image.new("RGB", (1400, 900), "#F5F1E8")
    draw = ImageDraw.Draw(image)
    title_font = pick_font(34)
    body_font = pick_font(24)
    small_font = pick_font(20)

    draw.rectangle((40, 40, 1360, 860), outline="#8B7355", width=8)
    draw.text((70, 65), "Demo Shelf - Beverage Set", fill="#4A3728", font=title_font)

    shelf_y = 250
    draw.rectangle((80, shelf_y, 1320, shelf_y + 28), fill="#A67C52")
    draw.rectangle((80, shelf_y + 300, 1320, shelf_y + 328), fill="#A67C52")

    box_width = 240
    gap = 55
    start_x = 120
    box_top = 100

    for idx, product in enumerate(products):
        x0 = start_x + idx * (box_width + gap)
        y0 = box_top + (idx % 2) * 40
        x1 = x0 + box_width
        y1 = y0 + 430
        color = CARD_COLORS[idx % len(CARD_COLORS)]

        draw.rounded_rectangle((x0, y0, x1, y1), radius=20, fill=color, outline="#2B2B2B", width=4)
        draw.rectangle((x0 + 20, y1 - 95, x1 - 20, y1 - 30), fill="#FFF7E6", outline="#2B2B2B", width=2)

        name_lines = wrap_text(draw, product["name"], body_font, box_width - 36)
        text_y = y0 + 45
        for line in name_lines[:3]:
            draw.text((x0 + 18, text_y), line, fill="white", font=body_font)
            text_y += 34

        draw.text((x0 + 18, text_y + 10), f'Brand: {product["brand"]}', fill="white", font=small_font)
        draw.text((x0 + 18, text_y + 44), f'Size: {product["size"]}', fill="white", font=small_font)
        draw.text((x0 + 18, text_y + 78), f'Category: {product["category"]}', fill="white", font=small_font)
        draw.text((x0 + 45, y1 - 82), f'${product["price"]:.2f}', fill="#2B2B2B", font=title_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple demo shelf image for Part B.")
    parser.add_argument(
        "--products-file",
        type=Path,
        default=Path(__file__).with_name("demo_products.json"),
        help="JSON file containing the product list.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).with_name("demo_shelf.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = load_products(args.products_file)
    build_demo_shelf(products, args.output_file)
    print(args.output_file)


if __name__ == "__main__":
    main()

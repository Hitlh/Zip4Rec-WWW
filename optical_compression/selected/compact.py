"""Selected CD compact strategy for published experiment reproduction; no agent calls."""
from functools import lru_cache
from pathlib import Path
from typing import Any
import math
from PIL import Image, ImageDraw, ImageFont

@lru_cache(maxsize=16)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(__file__).parents[1] / "fonts" / name), size)


def wrapped(draw: ImageDraw.ImageDraw, value: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = value.replace("\n", " \n ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if word == "\\n":
            lines.append(current)
            current = ""
            continue
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=fnt) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def pack_profile_paragraphs(value: str) -> list[str]:
    compact_prefixes = ("Preferred eras:", "Moods/styles:", "Diversity:", "Evidence strength:", "Uncertainties:")
    output: list[str] = []
    pending: list[str] = []
    for raw in value.splitlines():
        line = raw.strip()
        if line.startswith(compact_prefixes):
            pending.append(line)
            continue
        if pending:
            output.append(" • ".join(pending))
            pending.clear()
        if line:
            output.append(line)
    if pending:
        output.append(" • ".join(pending))
    return output


def pack_profile_fields(value: str) -> str:
    return " ".join(pack_profile_paragraphs(value))


def compact_sections(task: dict[str, Any]) -> dict[str, list[str]]:
    """Exact displayed section text, shared by the original renderer and readback audit."""
    item = task['item']
    return dict(user=[pack_profile_fields(task['profile'])], item=[
        f"Title: {item['title']}",
        " • ".join([
            f"Artists: {', '.join(item['artists']) or 'Unknown'}",
            f"Genres: {', '.join(item['genres']) or 'Unknown'}",
            f"Era / styles: {'; '.join([item['era'], ', '.join(item['moods'])]).strip('; ')}",
        ]),
        f"Summary: {item['summary']}",
    ])


def render_compact(task: dict[str, Any]) -> dict[str, Any]:
    cfg = {}  # Preserve original compact defaults (16px bottom pad, height aligned to 32).
    width, pad, body_size = 640, 10, 14
    title_font, section_font, body_font = font(18, True), font(15, True), font(body_size)
    probe = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(probe)
    sections = compact_sections(task)
    profile_lines = wrapped(draw, sections['user'][0], body_font, width - 2 * pad)
    details = sections['item']
    item_lines = [line for value in details for line in wrapped(draw, value, body_font, width - 2 * pad)]
    line_h = 16
    raw_height = 20 + 24 + len(profile_lines) * line_h + 8 + 24 + len(item_lines) * line_h + int(cfg.get("bottom_pad", 16))
    height = max(32, raw_height if not cfg.get("align_height", True) else math.ceil(raw_height / 32) * 32)
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, width - 4, image.height - 4), fill="white", outline="#c8d0dc", width=1)
    y = 12
    draw.text((pad, y), "PROFILE-CONDITIONED RELEVANCE", font=title_font, fill="#172033")
    y += 24
    draw.rectangle((7, y, width - 7, y + 19), fill="#dbeafe")
    draw.text((pad, y + 1), "USER PROFILE", font=section_font, fill="#153e75")
    y += 24
    for line in profile_lines:
        draw.text((pad, y), line, font=body_font, fill="#202938")
        y += line_h
    y += 4
    draw.rectangle((7, y, width - 7, y + 19), fill="#dcfce7")
    draw.text((pad, y + 1), "CANDIDATE ITEM", font=section_font, fill="#166534")
    y += 24
    for line in item_lines:
        draw.text((pad, y), line, font=body_font, fill="#202938")
        y += line_h
    output_path = task.get("path")
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, optimize=True)
    return {
        "sample_id": task["sample_id"],
        "width": image.width,
        "height": image.height,
        "estimated_visual_tokens": (image.width // 32) * (image.height // 32),
    }

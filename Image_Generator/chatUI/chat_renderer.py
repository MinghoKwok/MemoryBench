"""Chat UI renderer that embeds real face avatars.

Refactored from generate_chat_ui_dataset.py: instead of drawing geometric
circles for avatars, this renderer loads a face image per speaker and
composites it (cropped + circular-masked) into each message bubble's
avatar slot.

Public API:
    render_screenshot(messages, output_path, title, faces_root, subtitle="")

Where ``messages`` is a list of dicts:
    {
        "speaker_id": "P01",          # used to look up the face PNG
        "speaker_name": "Marcus",     # rendered above the bubble
        "side": "left" | "right",
        "text": "...",
        "kind": "text" | "image_card",
    }

The same face_path resolves to byte-identical avatar pixels every time
the same speaker is rendered, so a "lookup face" query attached to a
later QA can match in pixel space.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


# ----- Canvas constants (kept consistent with the legacy chat UI) -----
CANVAS_WIDTH = 768
CANVAS_HEIGHT = 1280
HEADER_HEIGHT = 104
BOTTOM_BAR_HEIGHT = 88
# Avatars need to survive low-detail OpenAI tiling (~85px) and Gemini's
# internal resize, so we render them large enough that even after the API's
# downsampling pipeline the face is still recognizable.
AVATAR_SIZE = 128
MESSAGE_GAP = 26
PADDING_X = 28
TEXT_LINE_HEIGHT = 22
MAX_BUBBLE_WIDTH = 430

LEFT_BUBBLE = "#F2F3F7"
RIGHT_BUBBLE = "#CFE8FF"
BG_COLOR = "#E9EDF2"
HEADER_COLOR = "#FFFFFF"
TEXT_COLOR = "#1B1F23"
SUBTEXT_COLOR = "#5F6B7A"
OUTLINE_COLOR = "#D4DAE3"


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=64)
def _circular_avatar(face_path: str, size: int) -> Image.Image:
    """Load a face image and produce a square circular-masked RGBA crop.

    Cached so the SAME pixel buffer is reused for every screenshot that
    shows this speaker — and crucially, can also be saved out as the
    canonical "query face" attached to a face-lookup QA.
    """
    src = Image.open(face_path).convert("RGB")
    # Center-crop to square first to avoid distortion
    side = min(src.size)
    left = (src.width - side) // 2
    top = (src.height - side) // 2
    src = src.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(src, (0, 0), mask)
    return out


def export_avatar_png(face_path: str, dest_path: Path, size: int = AVATAR_SIZE) -> Path:
    """Save the canonical circular avatar crop to disk.

    Used so the dialog JSON can attach this exact PNG as a question_image
    on face-lookup QAs — guaranteeing pixel-identity with the in-chat avatar.
    """
    avatar = _circular_avatar(str(face_path), size)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    avatar.save(dest_path, format="PNG")
    return dest_path


# ----- Geometry helpers -----

def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    lines: List[str] = []
    current: List[str] = []
    for word in text.split():
        trial = " ".join(current + [word])
        if current and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _bubble_box(side: str, top: int, content_width: int, line_count: int) -> Tuple[int, int, int, int]:
    bubble_width = min(MAX_BUBBLE_WIDTH, max(140, content_width + 34))
    bubble_height = max(60, 28 + line_count * TEXT_LINE_HEIGHT + 18)
    if side == "left":
        left = PADDING_X + AVATAR_SIZE + 18
        right = left + bubble_width
    else:
        right = CANVAS_WIDTH - PADDING_X - AVATAR_SIZE - 18
        left = right - bubble_width
    return (left, top, right, top + bubble_height)


def _avatar_box(side: str, top: int) -> Tuple[int, int, int, int]:
    if side == "left":
        left = PADDING_X
    else:
        left = CANVAS_WIDTH - PADDING_X - AVATAR_SIZE
    return (left, top, left + AVATAR_SIZE, top + AVATAR_SIZE)


def _draw_message(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    msg: Dict,
    top: int,
    name_font: ImageFont.ImageFont,
    text_font: ImageFont.ImageFont,
) -> int:
    side = msg.get("side", "left")
    text = str(msg.get("text", ""))
    face_path = msg["face_path"]
    speaker_name = msg.get("speaker_name", "")

    content_width = draw.textbbox((0, 0), text, font=text_font)[2]
    lines = _wrap_lines(draw, text, text_font, MAX_BUBBLE_WIDTH - 32)
    bubble = _bubble_box(side, top + 22, content_width, len(lines))
    avatar_pos = _avatar_box(side, top + 12)

    avatar = _circular_avatar(str(face_path), AVATAR_SIZE)
    canvas.paste(avatar, (avatar_pos[0], avatar_pos[1]), avatar)

    name_x = bubble[0] + 14
    name_y = top + 2
    draw.text((name_x, name_y), speaker_name, fill=SUBTEXT_COLOR, font=name_font)

    bubble_fill = LEFT_BUBBLE if side == "left" else RIGHT_BUBBLE
    draw.rounded_rectangle(bubble, radius=24, fill=bubble_fill, outline=OUTLINE_COLOR, width=2)
    for idx, line in enumerate(lines):
        draw.text(
            (bubble[0] + 16, bubble[1] + 14 + idx * TEXT_LINE_HEIGHT),
            line,
            fill=TEXT_COLOR,
            font=text_font,
        )
    bottom = bubble[1] + 14 + len(lines) * TEXT_LINE_HEIGHT + 16

    if msg.get("kind") == "image_card":
        card_top = bottom + 10
        card_height = 132
        draw.rounded_rectangle(
            (bubble[0], card_top, bubble[2], card_top + card_height),
            radius=24,
            fill="#FFF6D8",
            outline="#E8D29A",
            width=2,
        )
        draw.rectangle((bubble[0] + 18, card_top + 18, bubble[2] - 18, card_top + 84), fill="#F4C95D")
        draw.text((bubble[0] + 20, card_top + 96), "[ photo card attached ]", fill="#7A5E00", font=name_font)
        bottom = card_top + card_height

    return bottom + MESSAGE_GAP


def render_screenshot(
    messages: Sequence[Dict],
    output_path: Path,
    title: str,
    subtitle: str = "Synthetic group chat",
) -> Path:
    """Render a chat screenshot embedding face avatars.

    ``messages`` items must include speaker_id, speaker_name, side,
    face_path, text, and optionally kind.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    title_font = _load_font(28)
    sub_font = _load_font(18)
    name_font = _load_font(18)
    text_font = _load_font(21)

    # Header
    draw.rectangle((0, 0, CANVAS_WIDTH, HEADER_HEIGHT), fill=HEADER_COLOR, outline=OUTLINE_COLOR)
    draw.text((32, 24), title, fill=TEXT_COLOR, font=title_font)
    draw.text((32, 64), subtitle, fill=SUBTEXT_COLOR, font=sub_font)

    y = HEADER_HEIGHT + 24
    for msg in messages:
        y = _draw_message(img, draw, msg, y, name_font, text_font)
        if y > CANVAS_HEIGHT - BOTTOM_BAR_HEIGHT - 60:
            break

    # Bottom message bar
    draw.rectangle(
        (0, CANVAS_HEIGHT - BOTTOM_BAR_HEIGHT, CANVAS_WIDTH, CANVAS_HEIGHT),
        fill=HEADER_COLOR,
        outline=OUTLINE_COLOR,
    )
    draw.rounded_rectangle(
        (28, CANVAS_HEIGHT - 64, CANVAS_WIDTH - 28, CANVAS_HEIGHT - 24),
        radius=20,
        fill="#F6F8FB",
        outline=OUTLINE_COLOR,
    )
    draw.text((48, CANVAS_HEIGHT - 55), "Message", fill=SUBTEXT_COLOR, font=name_font)

    img.save(output_path)
    return output_path

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIALOG_PATH = REPO_ROOT / "Benchmark_Pipeline" / "data" / "dialog" / "Chat_UI_Memory_Test.json"
DEFAULT_IMAGE_ROOT = REPO_ROOT / "Benchmark_Pipeline" / "data" / "image" / "Chat_UI_Memory_Test"
DEFAULT_EPISODE_ROOT = Path(__file__).resolve().parent / "outputs"

CANVAS_WIDTH = 768
CANVAS_HEIGHT = 1280
HEADER_HEIGHT = 104
BOTTOM_BAR_HEIGHT = 88
AVATAR_SIZE = 56
MESSAGE_GAP = 18
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

FIRST_NAMES = [
    "Mia",
    "Ava",
    "Noah",
    "Liam",
    "Ella",
    "Zoe",
    "Nina",
    "Owen",
    "Leah",
    "Milo",
    "Ivy",
    "Theo",
]

TOPIC_SENTENCES: Dict[str, List[str]] = {
    "dinner": [
        "Should we lock dinner for Friday night?",
        "I can book dinner near the river.",
        "Dinner at 7 still works for me.",
    ],
    "meeting": [
        "The meeting got moved to 10:30.",
        "I sent the meeting notes already.",
        "Can we keep the meeting short tomorrow?",
    ],
    "movie": [
        "That movie trailer looked surprisingly good.",
        "Movie night is still on if everyone is free.",
        "I can grab the movie tickets online.",
    ],
    "airport": [
        "I just got to the airport security line.",
        "The airport is packed this morning.",
        "I am still twenty minutes from the airport.",
    ],
    "friday": [
        "Friday is the easiest day for me.",
        "I will be out of town until Friday afternoon.",
        "Friday after lunch should be safe.",
    ],
    "way": [
        "I am on my way now.",
        "On my way, five minutes out.",
        "I am on my way with the charger.",
    ],
    "photo": [
        "Uploading the photo card in a second.",
        "The photo card is right below this message.",
        "I added the photo so you can compare layouts.",
    ],
}

FILLER_SENTENCES = [
    "That works for me.",
    "Give me a minute.",
    "I will confirm soon.",
    "Thanks for the heads up.",
    "Let me check the details.",
    "I saw that earlier.",
    "We should keep the same plan.",
    "I can handle that part.",
]

QUERY_TYPES = (
    "speaker_binding",
    "cross_screenshot_consistency",
    "visual_structure",
    "hallucination_resistance",
)


@dataclass
class Speaker:
    speaker_id: str
    name: str
    avatar_id: str
    side: str
    fill: str
    accent: str
    pattern: str


@dataclass
class Message:
    message_id: str
    speaker_id: str
    text: str
    topic: str
    screenshot_index: int
    kind: str = "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic chat UI screenshots and benchmark JSON.")
    parser.add_argument("--num-episodes", type=int, default=8, help="Number of synthetic episodes to generate.")
    parser.add_argument("--num-speakers", type=int, default=4, help="Speakers per episode.")
    parser.add_argument("--screenshots-per-episode", type=int, default=3, help="Number of screenshots per episode.")
    parser.add_argument("--messages-per-screenshot", type=int, default=5, help="Messages rendered into each screenshot.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--dialog-out", type=Path, default=DEFAULT_DIALOG_PATH, help="Benchmark dialog JSON output.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Rendered image output root.")
    parser.add_argument(
        "--episode-out",
        type=Path,
        default=DEFAULT_EPISODE_ROOT,
        help="Episode-level metadata output directory.",
    )
    return parser.parse_args()


def make_speakers(rng: random.Random, count: int) -> List[Speaker]:
    palette = [
        ("#FFB4A2", "#C75146", "dot"),
        ("#A0C4FF", "#285185", "stripe"),
        ("#B9FBC0", "#237A57", "ring"),
        ("#FFD166", "#9B6A00", "cross"),
        ("#CDB4DB", "#6B3F8A", "dot"),
        ("#9BF6FF", "#21758A", "stripe"),
    ]
    names = rng.sample(FIRST_NAMES, k=count)
    speakers: List[Speaker] = []
    for idx in range(count):
        fill, accent, pattern = palette[idx % len(palette)]
        side = "left" if idx % 2 == 0 else "right"
        speakers.append(
            Speaker(
                speaker_id=f"spk_{idx + 1}",
                name=names[idx],
                avatar_id=f"avatar_{idx + 1}",
                side=side,
                fill=fill,
                accent=accent,
                pattern=pattern,
            )
        )
    return speakers


def make_messages(
    rng: random.Random,
    speakers: Sequence[Speaker],
    screenshots_per_episode: int,
    messages_per_screenshot: int,
) -> Tuple[List[Message], Dict[str, str]]:
    messages: List[Message] = []
    friday_speaker_id = rng.choice(speakers).speaker_id
    target_topics = {
        "dinner": rng.choice(speakers).speaker_id,
        "meeting": rng.choice(speakers).speaker_id,
        "friday": friday_speaker_id,
        "way": friday_speaker_id,
        "photo": rng.choice(speakers).speaker_id,
    }
    message_index = 1
    total_messages = screenshots_per_episode * messages_per_screenshot
    scheduled: List[Tuple[str, str, int]] = [
        ("dinner", target_topics["dinner"], 0),
        ("meeting", target_topics["meeting"], 0),
        ("friday", target_topics["friday"], min(2, screenshots_per_episode - 1)),
        ("way", target_topics["way"], min(1, screenshots_per_episode - 1)),
        ("photo", target_topics["photo"], min(1, screenshots_per_episode - 1)),
    ]

    occupied_slots: Dict[Tuple[int, int], Tuple[str, str]] = {}
    for topic, speaker_id, screenshot_index in scheduled:
        local_slot = len([1 for key in occupied_slots if key[0] == screenshot_index])
        occupied_slots[(screenshot_index, local_slot)] = (topic, speaker_id)

    for screenshot_index in range(screenshots_per_episode):
        for local_index in range(messages_per_screenshot):
            scheduled_fact = occupied_slots.get((screenshot_index, local_index))
            if scheduled_fact is not None:
                topic, speaker_id = scheduled_fact
                text = rng.choice(TOPIC_SENTENCES[topic])
                kind = "image_card" if topic == "photo" else "text"
            else:
                speaker_id = rng.choice(speakers).speaker_id
                topic = "filler"
                text = rng.choice(FILLER_SENTENCES)
                kind = "text"

            messages.append(
                Message(
                    message_id=f"m_{message_index:03d}",
                    speaker_id=speaker_id,
                    text=text,
                    topic=topic,
                    screenshot_index=screenshot_index,
                    kind=kind,
                )
            )
            message_index += 1

    while len(messages) < total_messages:
        speaker_id = rng.choice(speakers).speaker_id
        screenshot_index = len(messages) // messages_per_screenshot
        messages.append(
            Message(
                message_id=f"m_{len(messages) + 1:03d}",
                speaker_id=speaker_id,
                text=rng.choice(FILLER_SENTENCES),
                topic="filler",
                screenshot_index=screenshot_index,
                kind="text",
            )
        )
    return messages, target_topics


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def avatar_bbox(side: str, top: int, bubble_width: int, bubble_left: int, bubble_right: int) -> Tuple[int, int, int, int]:
    if side == "left":
        left = PADDING_X
    else:
        left = CANVAS_WIDTH - PADDING_X - AVATAR_SIZE
    return (left, top, left + AVATAR_SIZE, top + AVATAR_SIZE)


def wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> List[str]:
    lines: List[str] = []
    words = text.split()
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        trial_width = draw.textbbox((0, 0), trial, font=font)[2]
        if current and trial_width > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def bubble_bbox(side: str, top: int, content_width: int, line_count: int) -> Tuple[int, int, int, int]:
    bubble_width = min(MAX_BUBBLE_WIDTH, max(128, content_width + 34))
    bubble_height = max(56, 28 + line_count * TEXT_LINE_HEIGHT + 18)
    if side == "left":
        left = PADDING_X + AVATAR_SIZE + 16
        right = left + bubble_width
    else:
        right = CANVAS_WIDTH - PADDING_X - AVATAR_SIZE - 16
        left = right - bubble_width
    return (left, top, right, top + bubble_height)


def draw_avatar(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], speaker: Speaker) -> None:
    draw.ellipse(bbox, fill=speaker.fill, outline=speaker.accent, width=3)
    left, top, right, bottom = bbox
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    if speaker.pattern == "dot":
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=speaker.accent)
    elif speaker.pattern == "stripe":
        draw.line((left + 10, bottom - 12, right - 10, top + 12), fill=speaker.accent, width=5)
    elif speaker.pattern == "ring":
        draw.ellipse((left + 12, top + 12, right - 12, bottom - 12), outline=speaker.accent, width=4)
    else:
        draw.line((cx, top + 10, cx, bottom - 10), fill=speaker.accent, width=4)
        draw.line((left + 10, cy, right - 10, cy), fill=speaker.accent, width=4)


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
) -> int:
    lines = wrap_text_lines(draw, text, font, max_width)
    x, y = xy
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * TEXT_LINE_HEIGHT), line, fill=fill, font=font)
    return len(lines)


def draw_message(
    draw: ImageDraw.ImageDraw,
    speaker: Speaker,
    message: Message,
    top: int,
    name_font: ImageFont.ImageFont,
    text_font: ImageFont.ImageFont,
) -> int:
    content_width = draw.textbbox((0, 0), message.text, font=text_font)[2]
    lines = wrap_text_lines(draw, message.text, text_font, MAX_BUBBLE_WIDTH - 32)
    bubble = bubble_bbox(speaker.side, top + 18, content_width, len(lines))
    avatar = avatar_bbox(speaker.side, top + 8, bubble[2] - bubble[0], bubble[0], bubble[2])
    draw_avatar(draw, avatar, speaker)

    name_x = bubble[0] + 16
    name_y = top
    draw.text((name_x, name_y), speaker.name, fill=SUBTEXT_COLOR, font=name_font)

    bubble_fill = LEFT_BUBBLE if speaker.side == "left" else RIGHT_BUBBLE
    draw.rounded_rectangle(bubble, radius=24, fill=bubble_fill, outline=OUTLINE_COLOR, width=2)
    line_count = draw_wrapped_text(
        draw,
        (bubble[0] + 16, bubble[1] + 14),
        message.text,
        text_font,
        TEXT_COLOR,
        max_width=(bubble[2] - bubble[0] - 32),
    )
    bottom = bubble[1] + 14 + line_count * TEXT_LINE_HEIGHT + 16

    if message.kind == "image_card":
        card_top = bottom + 10
        card_height = 136
        draw.rounded_rectangle(
            (bubble[0], card_top, bubble[2], card_top + card_height),
            radius=24,
            fill="#FFF6D8",
            outline="#E8D29A",
            width=2,
        )
        draw.rectangle((bubble[0] + 18, card_top + 18, bubble[2] - 18, card_top + 84), fill="#F4C95D")
        draw.text((bubble[0] + 20, card_top + 96), "photo card attached", fill="#7A5E00", font=name_font)
        bottom = card_top + card_height

    return bottom + MESSAGE_GAP


def render_screenshot(
    speakers_by_id: Dict[str, Speaker],
    screenshot_messages: Sequence[Message],
    image_path: Path,
    title: str,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    title_font = load_font(28)
    name_font = load_font(18)
    text_font = load_font(21)

    draw.rectangle((0, 0, CANVAS_WIDTH, HEADER_HEIGHT), fill=HEADER_COLOR, outline=OUTLINE_COLOR)
    draw.text((32, 24), title, fill=TEXT_COLOR, font=title_font)
    draw.text((32, 60), "Synthetic group chat", fill=SUBTEXT_COLOR, font=name_font)

    y = HEADER_HEIGHT + 24
    for message in screenshot_messages:
        speaker = speakers_by_id[message.speaker_id]
        y = draw_message(draw, speaker, message, y, name_font, text_font)

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
    img.save(image_path)


def build_episode_payload(
    episode_id: str,
    speakers: Sequence[Speaker],
    messages: Sequence[Message],
    image_rel_paths: Sequence[str],
    queries: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "episode_id": episode_id,
        "speakers": [
            {
                "speaker_id": speaker.speaker_id,
                "name": speaker.name,
                "avatar_id": speaker.avatar_id,
                "side": speaker.side,
                "fill": speaker.fill,
                "accent": speaker.accent,
                "pattern": speaker.pattern,
            }
            for speaker in speakers
        ],
        "messages": [
            {
                "message_id": message.message_id,
                "speaker_id": message.speaker_id,
                "text": message.text,
                "topic": message.topic,
                "screenshot_index": message.screenshot_index,
                "kind": message.kind,
            }
            for message in messages
        ],
        "images": list(image_rel_paths),
        "queries": list(queries),
    }


def find_first_message(messages: Sequence[Message], topic: str) -> Message:
    for message in messages:
        if message.topic == topic:
            return message
    raise ValueError(f"Missing message for topic: {topic}")


def make_queries(
    episode_id: str,
    speakers: Sequence[Speaker],
    messages: Sequence[Message],
    target_topics: Dict[str, str],
) -> List[Dict[str, object]]:
    speakers_by_id = {speaker.speaker_id: speaker for speaker in speakers}
    dinner_speaker = speakers_by_id[target_topics["dinner"]]
    meeting_speaker = speakers_by_id[target_topics["meeting"]]
    friday_message = find_first_message(messages, "friday")
    friday_speaker = speakers_by_id[friday_message.speaker_id]
    way_message = find_first_message(messages, "way")
    photo_message = find_first_message(messages, "photo")
    photo_speaker = speakers_by_id[photo_message.speaker_id]

    return [
        {
            "sample_id": f"{episode_id}_q1",
            "episode_id": episode_id,
            "query_type": "speaker_binding",
            "query": "Who asked about dinner? Answer with the speaker name only.",
            "answer": dinner_speaker.name,
            "choices": [speaker.name for speaker in speakers],
            "point": [["X2"], ["Y1"]],
            "session_id": [episode_id],
            "clue": [f"{episode_id}:1"],
            "meta": {
                "target_topic": "dinner",
                "target_speaker_id": dinner_speaker.speaker_id,
                "target_speaker_name": dinner_speaker.name,
            },
        },
        {
            "sample_id": f"{episode_id}_q2",
            "episode_id": episode_id,
            "query_type": "cross_screenshot_consistency",
            "query": "The person who mentioned Friday later said what in another screenshot?",
            "answer": way_message.text,
            "choices": [message.text for message in messages if message.topic in {"friday", "way", "meeting", "dinner"}][:4],
            "point": [["X2"], ["Y2"]],
            "session_id": [episode_id],
            "clue": [f"{episode_id}:{friday_message.screenshot_index + 1}", f"{episode_id}:{way_message.screenshot_index + 1}"],
            "meta": {
                "reference_topic": "friday",
                "reference_avatar": friday_speaker.avatar_id,
            },
        },
        {
            "sample_id": f"{episode_id}_q3",
            "episode_id": episode_id,
            "query_type": "visual_structure",
            "query": f"Was the message '{way_message.text}' shown on the left or right side?",
            "answer": speakers_by_id[way_message.speaker_id].side,
            "choices": ["left", "right"],
            "point": [["X3"], ["Y1"]],
            "session_id": [episode_id],
            "clue": [f"{episode_id}:{way_message.screenshot_index + 1}"],
            "meta": {
                "target_message_id": way_message.message_id,
            },
        },
        {
            "sample_id": f"{episode_id}_q4",
            "episode_id": episode_id,
            "query_type": "hallucination_resistance",
            "query": f"Did {meeting_speaker.name} mention airport? Answer yes or no.",
            "answer": "no",
            "choices": ["yes", "no"],
            "point": [["X2"], ["Y2"]],
            "session_id": [episode_id],
            "clue": [f"{episode_id}:1", f"{episode_id}:2", f"{episode_id}:3"],
            "meta": {
                "distractor_topic": "airport",
                "probe_speaker_id": meeting_speaker.speaker_id,
                "probe_speaker_name": meeting_speaker.name,
            },
        },
        {
            "sample_id": f"{episode_id}_q5",
            "episode_id": episode_id,
            "query_type": "visual_structure",
            "query": "Who sent a message followed by a photo card? Answer with the speaker name only.",
            "answer": photo_speaker.name,
            "choices": [speaker.name for speaker in speakers],
            "point": [["X2", "X3"], ["Y1"]],
            "session_id": [episode_id],
            "clue": [f"{episode_id}:{photo_message.screenshot_index + 1}"],
            "meta": {
                "target_topic": "photo",
                "message_kind": "image_card",
                "target_speaker_id": photo_speaker.speaker_id,
                "target_speaker_name": photo_speaker.name,
            },
        },
    ]


def to_benchmark_dialog(
    episodes: Sequence[Dict[str, object]],
    dialog_out: Path,
    image_root: Path,
) -> Dict[str, object]:
    multi_session_dialogues: List[Dict[str, object]] = []
    qas: List[Dict[str, object]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        images = list(episode["images"])
        queries = list(episode["queries"])
        dialogues: List[Dict[str, object]] = []
        for index, image_rel in enumerate(images, start=1):
            dialogues.append(
                {
                    "round": f"{episode_id}:{index}",
                    "user": "Memorize this chat screenshot. Keep track of which avatar maps to which speaker and message.",
                    "assistant": "Stored for later memory questions.",
                    "image_id": [f"{episode_id}:{index}"],
                    "input_image": [f"../image/{image_root.name}/{Path(str(image_rel)).name}"],
                    "image_caption": [f"Synthetic chat UI screenshot {index} for {episode_id}."],
                }
            )

        multi_session_dialogues.append(
            {
                "session_id": episode_id,
                "date": "2026-03-19",
                "dialogues": dialogues,
            }
        )

        for query in queries:
            qas.append(
                {
                    "point": query["point"],
                    "question": query["query"],
                    "answer": query["answer"],
                    "session_id": query["session_id"],
                    "clue": query["clue"],
                    "query_type": query["query_type"],
                    "sample_id": query["sample_id"],
                    "meta": query["meta"],
                }
            )

    return {
        "character_profile": {
            "name": "MemEye Chat Curator",
            "persona_summary": "A synthetic curator focused on visual identity binding in chat interfaces.",
            "traits": ["systematic", "visual", "memory-oriented"],
            "conversation_style": "Short and factual. Emphasizes speaker-avatar-message consistency across screenshots.",
        },
        "multi_session_dialogues": multi_session_dialogues,
        "human-annotated QAs": qas,
    }


def generate_episode(
    rng: random.Random,
    episode_index: int,
    num_speakers: int,
    screenshots_per_episode: int,
    messages_per_screenshot: int,
    image_root: Path,
) -> Dict[str, object]:
    episode_id = f"CHAT_{episode_index:04d}"
    speakers = make_speakers(rng, num_speakers)
    messages, target_topics = make_messages(rng, speakers, screenshots_per_episode, messages_per_screenshot)
    speakers_by_id = {speaker.speaker_id: speaker for speaker in speakers}

    image_rel_paths: List[str] = []
    for screenshot_index in range(screenshots_per_episode):
        shot_messages = [message for message in messages if message.screenshot_index == screenshot_index]
        image_name = f"{episode_id}_{screenshot_index + 1}.png"
        render_screenshot(
            speakers_by_id,
            shot_messages,
            image_root / image_name,
            title=f"Weekend Plan {episode_index:02d}",
        )
        image_rel_paths.append(image_name)

    queries = make_queries(episode_id, speakers, messages, target_topics)
    return build_episode_payload(episode_id, speakers, messages, image_rel_paths, queries)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.image_root.mkdir(parents=True, exist_ok=True)
    args.episode_out.mkdir(parents=True, exist_ok=True)
    episodes: List[Dict[str, object]] = []

    for episode_index in range(1, args.num_episodes + 1):
        episodes.append(
            generate_episode(
                rng=rng,
                episode_index=episode_index,
                num_speakers=args.num_speakers,
                screenshots_per_episode=args.screenshots_per_episode,
                messages_per_screenshot=args.messages_per_screenshot,
                image_root=args.image_root,
            )
        )

    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_path = args.episode_out / f"{episode_id}.json"
        episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")

    benchmark_payload = to_benchmark_dialog(episodes, args.dialog_out, args.image_root)
    args.dialog_out.parent.mkdir(parents=True, exist_ok=True)
    args.dialog_out.write_text(json.dumps(benchmark_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "episodes": len(episodes),
        "images": len(episodes) * args.screenshots_per_episode,
        "qas": sum(len(episode["queries"]) for episode in episodes),
        "dialog_json": str(args.dialog_out),
        "image_root": str(args.image_root),
        "episode_root": str(args.episode_out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

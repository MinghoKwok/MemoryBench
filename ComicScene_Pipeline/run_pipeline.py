import argparse
import datetime as dt
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comic Memory Benchmark demo pipeline: crop panels, extract story, run VLM QA."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="ComicScene154/Data/Alley_Oop/images/Alley_Oop_Page_1.jpg",
        help="Path to source comic page image.",
    )
    parser.add_argument(
        "--json-a",
        type=str,
        default="ComicScene154/Data/Alley_Oop/benchmark_scenes/Alley_Oop_annotated_scenes_1.json",
        help="Path to annotated scenes JSON (with panels + summary).",
    )
    parser.add_argument(
        "--json-b",
        type=str,
        default="ComicScene154/Data/Alley_Oop/benchmark_refined_scenes/1/Alley_Oop_1_refined_output_1.json",
        help="Path to refined narrative JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory. Panels are saved to <output-dir>/panels.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for model generation.",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Only run Step 1 and Step 2, skip Step 3 model inference.",
    )
    parser.add_argument(
        "--single-image-test",
        action="store_true",
        help="Use only p1.jpg for VLM (debug mode).",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=360 * 360,
        help="Max pixels per panel for memory-safe multi-image inference.",
    )
    parser.add_argument(
        "--question-mode",
        type=str,
        choices=["open", "mcq", "both"],
        default="open",
        help="Question style: open (default), mcq, or both.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="",
        help="Optional JSON file containing a list of questions.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="output/results.json",
        help="Path to save evaluation results JSON.",
    )
    return parser.parse_args()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def parse_json_like(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    cleaned = strip_code_fences(value)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: grab the largest JSON object-like segment.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_first_page_from_json_a(data_a: Dict[str, Any]) -> Dict[str, Any]:
    # Support both shapes:
    # 1) {"comic_data": {"pages": [...]}}
    # 2) {"pages": [...]}
    pages: List[Dict[str, Any]] = []
    if isinstance(data_a.get("comic_data"), dict):
        pages = data_a["comic_data"].get("pages", [])
    if not pages:
        pages = data_a.get("pages", [])

    if not pages:
        raise ValueError("No pages found in JSON A.")
    return pages[0]


def ensure_image_path(image_path: Path, page_file_name: Optional[str], json_a_path: Path) -> Path:
    candidates: List[Path] = [image_path]

    if page_file_name:
        # Try relative to JSON A's dataset root.
        # json_a_path .../benchmark_scenes/file.json -> dataset_root is parent.parent
        dataset_root = json_a_path.parent.parent
        candidates.append(dataset_root / "images" / page_file_name)
        candidates.append(dataset_root / "output_images" / page_file_name)
        # Some files are stored without extension.
        if page_file_name.lower().endswith(".jpg"):
            candidates.append(dataset_root / "images" / page_file_name[:-4])
            candidates.append(dataset_root / "output_images" / page_file_name[:-4])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    all_candidates = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Image file not found. Tried:\n"
        f"{all_candidates}\n"
        "You can pass a valid --image-path explicitly."
    )


def safe_crop_bbox(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox length: {bbox}")
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_panels(image_path: Path, panels: List[Dict[str, Any]], output_panels_dir: Path) -> List[Path]:
    output_panels_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    with Image.open(image_path) as img:
        width, height = img.size
        for idx, panel in enumerate(panels, start=1):
            bbox = panel.get("bbox")
            if not isinstance(bbox, list):
                print(f"[WARN] panel {idx} missing bbox; skipped.")
                continue

            try:
                crop_box = safe_crop_bbox(bbox, width, height)
                panel_img = img.crop(crop_box)
            except Exception as exc:
                print(f"[WARN] panel {idx} crop failed: {exc}; skipped.")
                continue

            save_path = output_panels_dir / f"p{idx}.jpg"
            panel_img.save(save_path, format="JPEG")
            saved_paths.append(save_path)

    if not saved_paths:
        raise RuntimeError("No panel images were generated.")
    return saved_paths


def resize_for_memory_bench(panel_paths: List[Path], max_pixels: int) -> List[Path]:
    """
    Downsample each panel to control visual token usage in multi-image inference.
    Keeps independent image inputs to preserve sequential-memory testing.
    """
    if max_pixels <= 0:
        return panel_paths

    resized_paths: List[Path] = []
    for idx, panel_path in enumerate(panel_paths, start=1):
        out_path = panel_path.parent / f"lowres_p{idx}.jpg"
        with Image.open(panel_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            current_pixels = width * height
            if current_pixels > max_pixels:
                scale = (max_pixels / current_pixels) ** 0.5
                new_w = max(1, int(width * scale))
                new_h = max(1, int(height * scale))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img.save(out_path, format="JPEG", quality=90)
        resized_paths.append(out_path)
    return resized_paths


def extract_ground_truth_story(data_a: Dict[str, Any], data_b: Dict[str, Any]) -> str:
    descriptions: List[str] = []

    page = get_first_page_from_json_a(data_a)
    summary = page.get("summary")
    parsed_summary = parse_json_like(summary)

    if parsed_summary and isinstance(parsed_summary.get("Narrative_Arcs"), list):
        for arc in parsed_summary["Narrative_Arcs"]:
            if not isinstance(arc, dict):
                continue
            title = str(arc.get("title", "")).strip()
            desc = str(arc.get("description", "")).strip()
            if title or desc:
                descriptions.append(f"- {title}: {desc}".strip())

    if descriptions:
        return "\n".join(descriptions)

    # Fallback to JSON B hierarchy.
    major_arcs = data_b.get("Major_Arcs", [])
    for major in major_arcs if isinstance(major_arcs, list) else []:
        if not isinstance(major, dict):
            continue
        for sub_scene in major.get("sub_scenes", []) or []:
            if not isinstance(sub_scene, dict):
                continue
            title = str(sub_scene.get("title", "")).strip()
            desc = str(sub_scene.get("description", "")).strip()
            if title or desc:
                descriptions.append(f"- {title}: {desc}".strip())

    return "\n".join(descriptions) if descriptions else "No description could be extracted from JSON A/B."


def run_vlm_inference(
    model_path: str,
    panel_paths: List[Path],
    question: str,
    max_new_tokens: int,
    max_image_pixels: int,
) -> str:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        qwen_vl_utils = importlib.import_module("qwen_vl_utils")
        AutoProcessor = transformers.AutoProcessor
        AutoConfig = transformers.AutoConfig
        process_vision_info = qwen_vl_utils.process_vision_info
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies for VLM inference. Please install:\n"
            "pip install torch transformers qwen-vl-utils pillow"
        ) from exc

    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else "cpu"

    print(f"\n[INFO] Loading model: {model_path}")
    cfg = AutoConfig.from_pretrained(model_path)
    model_type = getattr(cfg, "model_type", "unknown")
    print(f"[INFO] Detected model_type: {model_type}")

    # In practice, qwen2_5_vl is often more stable in fp16 than bf16 on mixed CUDA stacks.
    if use_cuda:
        if model_type == "qwen2_5_vl":
            torch_dtype = torch.float16
        elif hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"[INFO] Device: {'cuda' if use_cuda else 'cpu'} | dtype: {torch_dtype}")

    model_cls = None
    if model_type == "qwen2_5_vl":
        model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    elif model_type == "qwen2_vl":
        model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration", None)

    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if model_cls is None:
        raise RuntimeError(
            f"Unsupported model_type '{model_type}' with current transformers build. "
            "Please upgrade transformers to a version supporting this Qwen VL model."
        )

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    # Avoid "invalid generation flags" warnings when using do_sample=False.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        for attr in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(model.generation_config, attr):
                setattr(model.generation_config, attr, None)

    print(f"[INFO] Resizing panels for memory-safe inference (max_pixels={max_image_pixels})...")
    safe_panel_paths = resize_for_memory_bench(panel_paths, max_pixels=max_image_pixels)
    content: List[Dict[str, str]] = [{"type": "image", "image": f"file://{p.resolve()}"} for p in safe_panel_paths]
    prompt = (
        "You are viewing a sequence of comic panels ordered chronologically from 1 to N.\n"
        "Hold earlier visual details in memory while reading later panels.\n"
        "Answer in clear English with 3-5 sentences based only on observable visual evidence.\n"
        f"Question: {question}"
    )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if not image_inputs:
        raise RuntimeError("No image inputs were parsed for Qwen-VL. Check panel paths.")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if use_cuda:
        inputs = inputs.to("cuda")

    def _decode_generated(generated_ids_tensor):
        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_tensor)
        ]
        trimmed_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (trimmed_text[0] if trimmed_text else "").strip()

    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    answer = _decode_generated(generated_ids)
    if answer:
        return answer

    # Retry once with conservative sampling if first pass is empty.
    with torch.inference_mode():
        generated_ids_retry = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, 128),
            min_new_tokens=8,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    retry_answer = _decode_generated(generated_ids_retry)
    if retry_answer:
        return retry_answer

    return "[EMPTY_VLM_ANSWER]"


def is_garbled_text(text: str) -> bool:
    t = text.strip()
    if len(t) < 25:
        return True
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in t)
    spaces = sum(ch.isspace() for ch in t)

    # Too many punctuation/symbols and too few letters -> likely degenerate output.
    letter_ratio = letters / max(len(t), 1)
    punct_ratio = punct / max(len(t), 1)
    if letter_ratio < 0.2 and punct_ratio > 0.35:
        return True

    # Repeated same symbol runs like "!!!!!!!!!!!!" are a strong signal.
    if re.search(r"([!?.,/\\\-_=+*#@~])\1{8,}", t):
        return True

    # Mostly numbers/symbols with almost no natural language spacing.
    if letters < 15 and (digits + punct) > max(30, spaces * 4):
        return True

    return False


def get_default_open_question() -> str:
    return (
        "Based on the visual narrative, describe what happens to the character "
        "'King Guz' (the one with the crown) from the first panel to the last panel. "
        "Does he stay on land?"
    )


def get_default_mcq_question() -> str:
    return (
        "Look at the sequence of 5 comic panels. Pay close attention to the character "
        "'King Guz' (wearing a crown) in Panel 1 and Panel 5.\n"
        "Question: What happens to his location?\n"
        "A) He stays on land the entire time.\n"
        "B) He starts in the water (drowning) in Panel 1, and is back in the water in Panel 5.\n"
        "C) He flies into the sky.\n"
        "Please answer with only the letter A, B, or C."
    )


def extract_mcq_choice(answer: str) -> str:
    text = answer.strip().upper()
    # Accept strict single-letter outputs first.
    if text in {"A", "B", "C"}:
        return text
    # Fallback: find a standalone choice token.
    match = re.search(r"\b([ABC])\b", text)
    if match:
        return match.group(1)
    return "INVALID"


def load_questions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.question_file:
        q_path = Path(args.question_file)
        data = load_json(q_path)
        items: List[Dict[str, Any]] = []
        raw_list = data.get("questions", data if isinstance(data, list) else [])
        if not isinstance(raw_list, list):
            raise ValueError("--question-file must be a JSON list or {'questions': [...]} format.")
        for idx, item in enumerate(raw_list, start=1):
            if isinstance(item, str):
                items.append({"id": f"q{idx}", "type": "custom", "text": item})
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                items.append(
                    {
                        "id": str(item.get("id", f"q{idx}")),
                        "type": str(item.get("type", "custom")),
                        "text": item["text"],
                        "gt": item.get("gt"),
                    }
                )
        if not items:
            raise ValueError("No valid questions loaded from --question-file.")
        return items

    if args.question_mode == "mcq":
        return [{"id": "mcq_default", "type": "mcq", "text": get_default_mcq_question(), "gt": "B"}]
    if args.question_mode == "both":
        return [
            {"id": "open_default", "type": "open", "text": get_default_open_question()},
            {"id": "mcq_default", "type": "mcq", "text": get_default_mcq_question(), "gt": "B"},
        ]
    return [{"id": "open_default", "type": "open", "text": get_default_open_question()}]


def main() -> None:
    args = parse_args()

    json_a_path = Path(args.json_a)
    json_b_path = Path(args.json_b)
    output_dir = Path(args.output_dir)
    output_panels_dir = output_dir / "panels"

    # Support both current repo layouts: with and without ComicScene154 prefix.
    if not json_a_path.exists():
        alt = Path(str(json_a_path).replace("ComicScene154/", "", 1))
        if alt.exists():
            json_a_path = alt
    if not json_b_path.exists():
        alt = Path(str(json_b_path).replace("ComicScene154/", "", 1))
        if alt.exists():
            json_b_path = alt

    data_a = load_json(json_a_path)
    data_b = load_json(json_b_path)
    first_page = get_first_page_from_json_a(data_a)

    page_file_name = first_page.get("file_name")
    image_path = ensure_image_path(Path(args.image_path), page_file_name, json_a_path)
    print(f"[INFO] Using image: {image_path}")
    print(f"[INFO] Using JSON A: {json_a_path}")
    print(f"[INFO] Using JSON B: {json_b_path}")

    # Step 1: crop panels
    panels = first_page.get("panels", [])
    if not isinstance(panels, list) or not panels:
        raise ValueError("No panels found in JSON A first page.")
    panel_paths = crop_panels(image_path=image_path, panels=panels, output_panels_dir=output_panels_dir)
    print(f"\n[Step 1] Saved {len(panel_paths)} panels to: {output_panels_dir}")
    for p in panel_paths:
        print(f"  - {p}")

    # Step 2: extract story context
    story_text = extract_ground_truth_story(data_a, data_b)
    print("\n[Step 2] Ground Truth Story:")
    print(story_text)

    # Step 3: VLM QA
    questions = load_questions(args)
    print(f"\n[Step 3] Loaded {len(questions)} question(s).")

    if args.skip_vlm:
        print("\n[INFO] --skip-vlm enabled. Step 3 was skipped.")
        return

    vlm_panel_paths = panel_paths[:1] if args.single_image_test else panel_paths
    if args.single_image_test:
        print("[INFO] --single-image-test enabled. Only using p1.jpg for VLM.")

    results: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions, start=1):
        question = q["text"]
        print(f"\n[Step 3.{idx}] Question ({q['type']}):")
        print(question)

        answer = run_vlm_inference(
            model_path=args.model_path,
            panel_paths=vlm_panel_paths,
            question=question,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
        )

        used_fallback = False
        if is_garbled_text(answer) and len(vlm_panel_paths) > 1:
            print("[WARN] Detected noisy VLM output. Retrying with single-image fallback.")
            fallback_question = (
                "Describe only what happens in this single comic panel image. "
                "Use clear English in 2-4 sentences."
            )
            answer = run_vlm_inference(
                model_path=args.model_path,
                panel_paths=[panel_paths[0]],
                question=fallback_question,
                max_new_tokens=min(args.max_new_tokens, 128),
                max_image_pixels=args.max_image_pixels,
            )
            used_fallback = True

        print("\n[VLM Answer]")
        print(answer)
        row: Dict[str, Any] = {
            "id": q["id"],
            "type": q["type"],
            "question": question,
            "answer": answer,
            "garbled": is_garbled_text(answer),
            "used_single_image_fallback": used_fallback,
        }
        if q["type"] == "mcq":
            pred = extract_mcq_choice(answer)
            gt = str(q.get("gt", "")).upper() if q.get("gt") else None
            row["mcq_pred"] = pred
            row["mcq_gt"] = gt
            row["mcq_valid"] = pred in {"A", "B", "C"}
            row["mcq_correct"] = bool(gt and pred == gt)
            print(f"[MCQ] pred={pred} gt={gt}")
        results.append(row)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "single_image_test": args.single_image_test,
        "max_new_tokens": args.max_new_tokens,
        "max_image_pixels": args.max_image_pixels,
        "story_text": story_text,
        "results": results,
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Saved results to: {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

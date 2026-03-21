from typing import Any, Dict, List

from .base import BaseRouter


class QwenLocalRouter(BaseRouter):
    def __init__(self, model_path: str, max_new_tokens: int = 128, system_prompt: str = "") -> None:
        import importlib

        self.torch = importlib.import_module("torch")
        self.transformers = importlib.import_module("transformers")
        self.qwen_vl_utils = importlib.import_module("qwen_vl_utils")

        self.max_new_tokens = max_new_tokens
        self.model_path = model_path
        self.system_prompt = system_prompt.strip() or (
            "You answer questions about a prior multimodal conversation. "
            "Use only the provided messages and images. Be concise and factual."
        )

        AutoConfig = self.transformers.AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        model_type = getattr(cfg, "model_type", "unknown")

        model_cls = None
        if model_type == "qwen2_5_vl":
            model_cls = getattr(self.transformers, "Qwen2_5_VLForConditionalGeneration", None)
        elif model_type == "qwen2_vl":
            model_cls = getattr(self.transformers, "Qwen2VLForConditionalGeneration", None)
        if model_cls is None:
            model_cls = getattr(self.transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            raise RuntimeError(f"Unsupported model_type: {model_type}")

        use_cuda = self.torch.cuda.is_available()
        if use_cuda and hasattr(self.torch.cuda, "is_bf16_supported") and self.torch.cuda.is_bf16_supported():
            dtype = self.torch.bfloat16
        elif use_cuda:
            dtype = self.torch.float16
        else:
            dtype = self.torch.float32

        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else "cpu",
        )
        self.model.eval()
        self.processor = self.transformers.AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.process_vision_info = self.qwen_vl_utils.process_vision_info
        self.use_cuda = use_cuda
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            for attr in ("temperature", "top_p", "top_k", "typical_p"):
                if hasattr(self.model.generation_config, attr):
                    setattr(self.model.generation_config, attr, None)

    def _to_qwen_messages(self, history_messages: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]
        for msg in history_messages:
            content: List[Dict[str, str]] = []
            for img in msg.get("images", []) or []:
                content.append({"type": "image", "image": f"file://{img}"})
            txt = msg.get("text", "")
            if txt:
                content.append({"type": "text", "text": txt})
            if content:
                messages.append({"role": msg.get("role", "user"), "content": content})

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Answer based on the conversation and images above. "
                            "Be concise and factual.\n"
                            f"Question: {question}"
                        ),
                    }
                ],
            }
        )
        return messages

    def answer(self, history_messages: List[Dict[str, Any]], question: str) -> str:
        messages = self._to_qwen_messages(history_messages, question)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if self.use_cuda:
            inputs = inputs.to("cuda")

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                max_time=25,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (out[0] if out else "").strip()

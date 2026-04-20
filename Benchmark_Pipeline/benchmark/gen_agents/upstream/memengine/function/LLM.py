from abc import ABC, abstractmethod

import json

from openai import OpenAI

class BaseLLM(ABC):
    """
    Provides a convenient interface to utilize the powerful capability of different large language models.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def fast_run(self, query):
        pass

class APILLM(BaseLLM):
    """
    Utilize LLM from APIs.
    """
    def __init__(self, config):
        super().__init__(config)

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content,
            'input_token': response.usage.prompt_tokens,
            'output_token': response.usage.completion_tokens
        }

    def _extract_text(self, response):
        message = response.choices[0].message
        content = getattr(message, 'content', None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get('text')
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return '\n'.join(parts)
        return ''

    def run(self, message_list):
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=message_list,
            temperature=self.config.temperature
        )
        response = self.parse_response(response)
        return response

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response['result']

    def fast_run_json_items(self, query, item_count, item_kind='item'):
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return a JSON object that strictly matches the requested schema. "
                        "Do not add extra keys."
                    ),
                },
                {"role": "user", "content": query},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": f"{item_kind}_list_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": item_count,
                                "maxItems": item_count,
                            }
                        },
                        "required": ["items"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=self.config.temperature,
        )
        raw = self._extract_text(response).strip()
        payload = json.loads(raw)
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Structured {item_kind} response missing items list")
        normalized = [str(item).strip() for item in items if str(item).strip()]
        if len(normalized) != item_count:
            raise ValueError(
                f"Structured {item_kind} response returned {len(normalized)} items, expected {item_count}"
            )
        return normalized

class LocalVLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)

        from vllm import LLM, SamplingParams
        self.model = LLM(config.name)

        self.sampling_params = SamplingParams(temperature=config.temperature)
    
    def run(self, message_list):
        return self.model.chat(message_list,self.sampling_params)[0]
    
    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response.outputs[0].text

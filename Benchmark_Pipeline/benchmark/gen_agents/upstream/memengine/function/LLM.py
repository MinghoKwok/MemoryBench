from abc import ABC, abstractmethod

import json
import re

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

    def _provider_name(self):
        provider = str(getattr(self.config, 'provider', '') or '').strip().lower()
        if provider:
            return provider
        base_url = str(getattr(self.config, 'base_url', '') or '').strip().lower()
        if 'generativelanguage.googleapis.com' in base_url:
            return 'gemini_api'
        return 'openai_api'

    def _supports_json_schema(self):
        return self._provider_name() == 'openai_api'

    def parse_response(self, response):
        usage = getattr(response, 'usage', None)
        return {
            'run_id': getattr(response, 'id', ''),
            'time_stamp': getattr(response, 'created', None),
            'result': response.choices[0].message.content,
            'input_token': getattr(usage, 'prompt_tokens', 0),
            'output_token': getattr(usage, 'completion_tokens', 0)
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

    def _strip_json_fences(self, raw):
        text = str(raw or '').strip()
        if text.startswith('```'):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines).strip()
        return text

    def _parse_json_payload(self, raw):
        text = self._strip_json_fences(raw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            raise

    def _create_with_optional_schema(self, *, messages, schema_name, schema, temperature):
        request = {
            'model': self.config.name,
            'messages': messages,
            'temperature': temperature,
        }
        if self._supports_json_schema():
            request['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': schema_name,
                    'schema': schema,
                    'strict': True,
                },
            }
        return self.client.chat.completions.create(**request)

    def _request_structured_payload(
        self,
        *,
        messages,
        schema_name,
        schema,
        validate_payload,
        max_attempts=3,
    ):
        base_messages = list(messages)
        last_error = None
        last_raw = ''
        for attempt in range(1, max_attempts + 1):
            response = self._create_with_optional_schema(
                messages=base_messages,
                schema_name=schema_name,
                schema=schema,
                temperature=self.config.temperature,
            )
            raw = self._extract_text(response).strip()
            last_raw = raw
            try:
                payload = self._parse_json_payload(raw)
                validate_payload(payload)
                return payload
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                repair_messages = list(messages)
                if raw:
                    repair_messages.append({"role": "assistant", "content": raw})
                repair_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid for the required schema. "
                            "Return only valid JSON with no markdown fences, no prose, and no extra keys."
                        ),
                    }
                )
                base_messages = repair_messages
        raise ValueError(
            f"Structured response failed after {max_attempts} attempt(s): {last_error}. Raw response: {last_raw}"
        )

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response['result']

    def fast_run_json_items(self, query, item_count, item_kind='item'):
        schema = {
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
        }
        def validate_payload(payload):
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError(f"Structured {item_kind} response missing items list")
            normalized = [str(item).strip() for item in items if str(item).strip()]
            if len(normalized) != item_count:
                raise ValueError(
                    f"Structured {item_kind} response returned {len(normalized)} items, expected {item_count}"
                )

        payload = self._request_structured_payload(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return a JSON object that strictly matches the requested schema. "
                        "Do not add extra keys. Do not wrap the JSON in markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{query}\n\nReturn exactly {item_count} {item_kind}(s) in JSON with this schema only: "
                        '{"items":["..."]}'
                    ),
                },
            ],
            schema_name=f"{item_kind}_list_response",
            schema=schema,
            validate_payload=validate_payload,
        )
        items = payload["items"]
        normalized = [str(item).strip() for item in items if str(item).strip()]
        return normalized

    def fast_run_number(self, query):
        raw = self.fast_run(query)
        text = str(raw).strip()
        match = re.search(r'-?\d+(?:\.\d+)?', text)
        if not match:
            raise ValueError(f"Numeric response missing number: {text}")
        return float(match.group(0))

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

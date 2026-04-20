from abc import ABC, abstractmethod
from memengine.function.LLM import *
from langchain.prompts import PromptTemplate

class BaseReflector(ABC):
    """
    Draw new insights in higher level from existing information, commonly for reflection and optimization operations.
    """
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def generate_insight(self, *args, **kwargs):
        pass

class GAReflector(BaseReflector):
    def __init__(self, config):
        super().__init__(config)

        self.accmulated_importance = 0
        self.recursion_list = []
        self.llm = eval(config.LLM_config.method)(config.LLM_config)

    def reset(self):
        self.accmulated_importance = 0
        self.recursion_list = []
    
    def add_reflection(self, importance, recursion):
        self.accmulated_importance += importance
        self.recursion_list.append(recursion)
    
    def delete_reflection(self, index):
        raise NotImplementedError()
    
    def get_recursion(self, index):
        return self.recursion_list[index]

    def get_current_accmulated_importance(self):
        return self.accmulated_importance
    
    def get_reflection_threshold(self):
        return self.config.threshold

    def _normalize_text_items(self, text):
        items = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            while line and (line[0].isdigit() or line[0] in {'-', '*', ')', '.', ':'}):
                line = line[1:].strip()
            items.append(line)
        return [item for item in items if item]

    def _generate_items(self, prompt, item_count, item_kind):
        attempts = [
            prompt,
            (
                prompt
                + f"\n\nReturn exactly {item_count} {item_kind}(s) as JSON with this schema only: "
                  '{"items":["..."]}'
            ),
        ]
        last_error = None
        for attempt_index, attempt_prompt in enumerate(attempts, start=1):
            try:
                if hasattr(self.llm, 'fast_run_json_items'):
                    items = self.llm.fast_run_json_items(attempt_prompt, item_count, item_kind=item_kind)
                else:
                    raw = self.llm.fast_run(attempt_prompt)
                    items = self._normalize_text_items(raw)
                    if len(items) != item_count:
                        raise ValueError(
                            f"Plain-text {item_kind} response returned {len(items)} items, expected {item_count}"
                        )
                return items
            except Exception as exc:
                last_error = exc
                if attempt_index < len(attempts):
                    print(
                        f"[WARN] gen_agents reflection {item_kind} generation malformed on attempt "
                        f"{attempt_index}/{len(attempts)}: {exc}. Retrying with stricter formatting."
                    )
        print(
            f"[WARN] gen_agents reflection {item_kind} generation skipped after "
            f"{len(attempts)} malformed attempt(s): {last_error}"
        )
        return []

    def self_ask(self, input_dict):
        prompt_template = PromptTemplate(
            input_variables=self.config.question_prompt.input_variables,
            template=self.config.question_prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        return self._generate_items(prompt, self.config.question_number, 'question')
    
    def generate_insight(self, input_dict):
        prompt_template = PromptTemplate(
            input_variables=self.config.insight_prompt.input_variables,
            template=self.config.insight_prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        return self._generate_items(prompt, self.config.insight_number, 'insight')

class TrialReflector(BaseReflector):
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)
    
    def generate_insight(self, input_dict):
        prompt_template = PromptTemplate(
            input_variables=self.config.prompt.input_variables,
            template=self.config.prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        res = self.llm.fast_run(prompt)

        return res

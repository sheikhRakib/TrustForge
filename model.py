from __future__ import annotations

import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MAX_INPUT_TOKENS = 32_768


class LLMModel:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.max_input_tokens = DEFAULT_MAX_INPUT_TOKENS
        self.call_count = 0
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LLMModel":
        """Load a model with ``device_map="auto"`` across visible GPUs."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required. Run inside a GPU allocation; CPU fallback is disabled."
            )
        print(f"Loading {model_name}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            dtype="auto",
            device_map="auto",
            max_memory={
                i: int(torch.cuda.get_device_properties(i).total_memory - 6 * 1024**3)
                for i in range(torch.cuda.device_count())
            },
        )

        placement = getattr(model, "hf_device_map", {})
        if any(str(d) in {"cpu", "disk"} for d in placement.values()):
            raise RuntimeError(
                "Model offloaded to CPU/disk; request more GPU memory."
            )
        model.eval()
        print(
            f"GPU placement: {placement}; attention=sdpa; dtype={model.dtype}",
            flush=True,
        )
        label = str(getattr(model, "device", "auto"))
        print(f"Loaded on {label}", flush=True)
        return cls(tokenizer, model)

    @property
    def device(self):
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    def _encode(self, system, user):
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

    def split_user(self, system, user, max_new_tokens=256):
        """Partition all user text while preserving the system and chat template.

        Prefer complete lines. Extremely long individual lines are token-split.
        Never silently truncate the input or the assistant generation marker.
        """
        context = getattr(
            self.model.config,
            "max_position_embeddings",
            self.max_input_tokens + max_new_tokens,
        )
        limit = min(self.max_input_tokens, context - max_new_tokens)
        overhead = self._encode(system, "").input_ids.shape[1] + 16
        budget = limit - overhead
        if budget < 64:
            raise ValueError(
                "Input budget too small for system instructions and output reserve"
            )
        if self._encode(system, user).input_ids.shape[1] <= limit:
            return [user]
        lines = user.splitlines(keepends=True)
        encoded = self.tokenizer(lines, add_special_tokens=False)["input_ids"]
        parts = []
        current = []
        size = 0
        for line, ids in zip(lines, encoded):
            if size + len(ids) > budget and current:
                parts.append("".join(current))
                current = []
                size = 0
            if len(ids) > budget:
                remaining = line
                while remaining:
                    low, high = 1, len(remaining)
                    while low < high:
                        middle = (low + high + 1) // 2
                        if (
                            len(
                                self.tokenizer(
                                    remaining[:middle], add_special_tokens=False
                                )["input_ids"]
                            )
                            <= budget
                        ):
                            low = middle
                        else:
                            high = middle - 1
                    parts.append(remaining[:low])
                    remaining = remaining[low:]
            else:
                current.append(line)
                size += len(ids)
        if current:
            parts.append("".join(current))
        # Token-boundary merges and template effects can change length on re-encoding.
        checked = []
        for part in parts:
            if self._encode(system, part).input_ids.shape[1] > limit:
                if len(part) < 2:
                    raise ValueError("Cannot split input within context budget")
                middle = len(part) // 2
                checked.extend(self.split_user(system, part[:middle], max_new_tokens))
                checked.extend(self.split_user(system, part[middle:], max_new_tokens))
            else:
                checked.append(part)
        print(
            f"  split oversized input into {len(checked)} complete review chunks",
            flush=True,
        )
        return checked

    def generate(self, system, user, max_new_tokens=256):
        """GPU inference with explicit context checks."""
        inputs = self._encode(system, user)
        input_len = inputs.input_ids.shape[1]
        context = getattr(
            self.model.config,
            "max_position_embeddings",
            self.max_input_tokens + max_new_tokens,
        )
        if input_len > self.max_input_tokens or input_len + max_new_tokens > context:
            raise ValueError(
                f"Input {input_len} exceeds budget; use split_user before generate"
            )
        with self._lock, torch.inference_mode():
            inputs = inputs.to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response_ids = outputs[0][input_len:]
            response = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            ).strip()
            self.call_count += 1
        return response

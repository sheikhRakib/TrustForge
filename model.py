import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMModel:
    """Thin wrapper around a local Hugging Face causal LM."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_pretrained(cls, model_name: str) -> "LLMModel":
        print(f"Loading {model_name}...", flush=True)
        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(
            "Tokenizer loaded. Loading model weights (this may take several minutes)...",
            flush=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        print("Loaded on", model.device, flush=True)
        return cls(tokenizer, model)

    @property
    def device(self):
        return self.model.device

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Run one chat completion and return the decoded response text."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return response.strip()

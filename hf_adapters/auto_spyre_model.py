import types

import torch
from transformers import (
    AutoConfig,
    GemmaConfig,
    Granite4VisionConfig,
    GraniteConfig,
    GraniteMoeHybridConfig,
    LlamaConfig,
    MistralConfig,
    Olmo2Config,
    OlmoConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
    SmolLM3Config,
)

from hf_adapters import (
    hf_gemma,
    hf_granite,
    hf_granite_vision,
    hf_granitemoehybrid,
    hf_llama,
    hf_mistral,
    hf_olmo,
    hf_olmo2,
    hf_phi3,
    hf_qwen2,
    hf_qwen3,
    hf_smollm3,
)
from hf_adapters.hf_common import load_model_common

CONFIG_TO_ADAPTER_MODULE_MAPPING = {
    GemmaConfig: hf_gemma,
    Granite4VisionConfig: hf_granite_vision,
    GraniteConfig: hf_granite,
    GraniteMoeHybridConfig: hf_granitemoehybrid,
    LlamaConfig: hf_llama,
    MistralConfig: hf_mistral,
    OlmoConfig: hf_olmo,
    Olmo2Config: hf_olmo2,
    Phi3Config: hf_phi3,
    Qwen2Config: hf_qwen2,
    Qwen3Config: hf_qwen3,
    SmolLM3Config: hf_smollm3,
}


class AutoSpyreModelForCausalLM:

    @staticmethod
    def from_pretrained(model_name_or_path, dtype=torch.float16):
        # Determine the appropriate Spyre adapter module for the model
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        if type(model_config) not in CONFIG_TO_ADAPTER_MODULE_MAPPING:
            raise Exception(
                f"Model {model_name_or_path} of type {type(model_config)} "
                "is not supported"
            )

        module = CONFIG_TO_ADAPTER_MODULE_MAPPING[type(model_config)]

        # Check if module has custom load_model function
        if hasattr(module, "load_model"):
            # Custom adapter loading method (e.g., Granite Vision)
            model = module.load_model(model_name_or_path, dtype)
        else:
            model = load_model_common(
                model_name_or_path, module.prepare_for_spyre, dtype
            )

        # Attach generate method using the module's forward function
        def model_generate(self, tokenizer, prompts, **kwargs):
            from hf_adapters.hf_common import generate

            return generate(module._run_forward, self, tokenizer, prompts, **kwargs)

        model.generate = types.MethodType(model_generate, model)

        return model

import logging
import tempfile
from typing import Optional

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class LoRAModelForEvaluation:
    """
    Load a base model together with saved LoRA adapters, merge them and return
    a standalone model for inference.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Load a model with saved LoRA adapters, merge the adapters into the base
        model and unload PEFT wrappers.

        Args:
            pretrained_model_name_or_path: Path to the saved LoRA adapters.
            base_model_name_or_path: Optional explicit base model path. When
                omitted, it is read from the adapter config.
            **kwargs: Additional arguments forwarded to the base model loader.
        """
        quantization_config = kwargs.pop("quantization_config", None)
        reload_quantized = quantization_config is not None
        adapter_path = pretrained_model_name_or_path
        # base_load_kwargs = dict(kwargs)

        logger.info(f"Loading LoRA adapter config from {adapter_path}")
        peft_config = PeftConfig.from_pretrained(adapter_path)

        base_model_path = base_model_name_or_path or peft_config.base_model_name_or_path
        if base_model_path is None:
            raise ValueError(
                "Base model path not provided and not found in adapter config. "
                "Please supply `base_model_name_or_path` or ensure the adapter "
                "config contains `base_model_name_or_path`."
            )

        logger.info(f"Loading base model from {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            # merge requires the base weights, so load without quantization
            #**base_load_kwargs,
        )

        logger.info(f"Loading LoRA adapters from {adapter_path}")
        lora_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )

        logger.info("Merging LoRA adapters into the base model")
        merged_model = lora_model.merge_and_unload()

        if reload_quantized:
            logger.info("Re-loading merged model with quantization applied")
            with tempfile.TemporaryDirectory() as tmpdir:
                merged_model.save_pretrained(tmpdir)
                merged_model = AutoModelForCausalLM.from_pretrained(
                    tmpdir,
                    quantization_config=quantization_config,
                    **kwargs,
                )
                logger.info(f"Model re-loaded with quantization {merged_model.config.quantization_config}")

        return merged_model
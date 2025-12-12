from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from omegaconf import DictConfig, open_dict, ListConfig
from typing import Optional
import torch
import logging
import os
from transformers import BitsAndBytesConfig

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)


def _get_bnb_config(quantization_config):
    """Map user-friendly quantization strings to BitsAndBytes configs."""
    if isinstance(quantization_config, BitsAndBytesConfig):
        return quantization_config
    if quantization_config == "qlora":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage_dtype=torch.bfloat16, 
        )
    if quantization_config == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    if quantization_config == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization_config is not None:
        logger.warning(
            f"Unknown quantization_config '{quantization_config}', loading without quantization."
        )
    return None


class LoRAModelForCausalLM:
    """
    Wrapper class for loading models with LoRA adapters.
    Supports the specified LoRA configuration parameters.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        lora_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        """
        Load a model with LoRA adapters.
        Args:
            pretrained_model_name_or_path: Path to the pretrained model
            lora_config: LoRA configuration parameters
            **kwargs: Additional arguments for model loading
        """
        bnb_config = kwargs.get("quantization_config", None)
        # Avoid HF auto device_map (can scatter modules across GPUs) when using bitsandbytes.
        # For DDP/DeepSpeed runs we instead pin the quantized model to the local rank device
        # so each process owns its shard on the correct GPU.
        if bnb_config is not None and "device_map" not in kwargs:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                kwargs["device_map"] = {"": local_rank}
            else:
                kwargs["device_map"] = None

        # Default LoRA configuration
        default_lora_config = {
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
                "lm_head",
            ],
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "r": 128,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Merge with provided config
        if lora_config:
            lora_params = dict(lora_config)
        else:
            lora_params = default_lora_config.copy()

        # Convert OmegaConf objects to regular Python types for JSON serialization
        def convert_omegaconf_to_python(obj):
            """Convert OmegaConf objects to regular Python types."""
            if isinstance(obj, ListConfig):
                return [convert_omegaconf_to_python(item) for item in obj]
            elif isinstance(obj, DictConfig):
                return {k: convert_omegaconf_to_python(v) for k, v in obj.items()}
            elif hasattr(obj, "_content"):  # Fallback for other OmegaConf types
                if isinstance(obj._content, list):
                    return [convert_omegaconf_to_python(item) for item in obj._content]
                elif isinstance(obj._content, dict):
                    return {
                        k: convert_omegaconf_to_python(v)
                        for k, v in obj._content.items()
                    }
                else:
                    return obj._content
            else:
                return obj

        # Convert all parameters to ensure JSON serialization compatibility
        lora_params = convert_omegaconf_to_python(lora_params)

        # Additional manual conversion to ensure all types are correct
        lora_params = {
            "target_modules": list(lora_params["target_modules"]),
            "lora_alpha": int(lora_params["lora_alpha"]),
            "lora_dropout": float(lora_params["lora_dropout"]),
            "r": int(lora_params["r"]),
            "bias": str(lora_params["bias"]),
            "task_type": str(lora_params["task_type"]),
        }

        # Log converted parameters for debugging
        logger.info(f"Converted LoRA parameters: {lora_params}")
        logger.info(f"target_modules type: {type(lora_params['target_modules'])}")
        logger.info(f"target_modules content: {lora_params['target_modules']}")

        # Test JSON serialization to ensure compatibility
        try:
            import json

            json.dumps(lora_params)
            logger.info("✅ LoRA parameters are JSON serializable")
        except Exception as e:
            logger.error(f"❌ LoRA parameters are NOT JSON serializable: {e}")
            raise ValueError(f"LoRA parameters cannot be serialized to JSON: {e}")

        # Load the base model
        logger.info(f"Loading base model from {pretrained_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if bnb_config is not None:
            base_model = prepare_model_for_kbit_training(base_model)

        # Create LoRA configuration with converted parameters
        peft_config = LoraConfig(
            target_modules=lora_params["target_modules"],
            lora_alpha=lora_params["lora_alpha"],
            lora_dropout=lora_params["lora_dropout"],
            r=lora_params["r"],
            bias=lora_params["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to the model
        logger.info(f"Applying LoRA with config: {peft_config}")
        model = get_peft_model(base_model, peft_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        return model


def get_lora_model(model_cfg: DictConfig):
    """
    Load a model with LoRA adapters using the model configuration.
    Args:
        model_cfg: Model configuration containing model_args, tokenizer_args, and lora_config
    Returns:
        Tuple of (model, tokenizer)
    """
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )

    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    lora_config = model_cfg.get("lora_config", None)
    quantization_config = model_cfg.get("quantization_config", None)

    # Get torch dtype using the same logic as the main module
    torch_dtype = get_dtype(model_args)

    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)

    try:
        bnb_config = _get_bnb_config(quantization_config)
        if bnb_config is not None:
            logger.info(
                f"Loading LoRA model {model_path} with quantization config: {bnb_config}"
            )
          
        model = LoRAModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path,
            lora_config=lora_config,
            torch_dtype=torch_dtype,
            cache_dir=hf_home,
            quantization_config=bnb_config,
            **model_args,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching LoRA model using LoRAModelForCausalLM.from_pretrained()."
        )

    # Load tokenizer using the same logic as the main module
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def get_dtype(model_args):
    """Extract torch dtype from model arguments."""
    with open_dict(model_args):
        torch_dtype_str = model_args.pop("torch_dtype", None)

    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype_str in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype_str}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )

    if torch_dtype_str is None:
        return torch.float32

    if torch_dtype_str == "bfloat16":
        return torch.bfloat16
    elif torch_dtype_str == "float16":
        return torch.float16
    elif torch_dtype_str == "float32":
        return torch.float32

    return torch.float32


def get_tokenizer(tokenizer_args):
    """Load tokenizer from tokenizer arguments."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_args.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_args}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")

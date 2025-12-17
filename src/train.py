import logging
import os
import sys

import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything
from transformers.utils import logging as transformers_logging

logger = logging.getLogger(__name__)
logging.getLogger("deepspeed").setLevel(logging.ERROR)


def _silence_non_main_process():
    """
    Prevent non-zero local ranks from spamming stdout/stderr.
    Accelerate/torchrun set LOCAL_RANK; we mute everything except rank 0.
    """
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        local_rank = 0

    if local_rank != 0:
        # Lower verbosity for libraries that log through the logging module.
        logging.disable(logging.CRITICAL)
        transformers_logging.set_verbosity_error()

        # Drop all stdout/stderr from non-main ranks so Slurm gets a single stream.
        null_stream = open(os.devnull, "w")
        sys.stdout = null_stream
        sys.stderr = null_stream


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    _silence_non_main_process()
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Keep checkpointing happy while leaving non-LoRA weights frozen
    if cfg.trainer.args.get("gradient_checkpointing", False):
        target = getattr(model, "base_model", model)
        if hasattr(target, "enable_input_require_grads"):
            print("Enabling input require grads for gradient checkpointing...")
            target.enable_input_require_grads()

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        logger.info(
            "\x1b[32mTraining setup: epochs=%s, batch_size=%s, learning_rate=%s, gradient_accumulation_steps=%s, weight_decay=%s\x1b[0m",
            trainer_args.num_train_epochs,
            trainer_args.per_device_train_batch_size,
            trainer_args.learning_rate,
            trainer_args.gradient_accumulation_steps,
            trainer_args.weight_decay,
        )
        trainer.train()
        trainer.accelerator.wait_for_everyone()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        print("Model saved.")

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # One-time sanity check: ensure only LoRA layers are trainable when LoRA is enabled.
    if getattr(model_cfg, "use_lora", False):
        from model.lora import summarize_lora_trainable_state

        print(summarize_lora_trainable_state(model))

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
        trainer.train()
        trainer.accelerator.wait_for_everyone()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        print("Model saved.")

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()

import os
import logging
import gc

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from models import build_model
from mt.ds import build_dataset
from utils.mt.argparser import parser

logging.basicConfig(level=logging.INFO)

wandb_project_name = "thesis"
wandb_entity = "hugo-pitorro"


def get_trainer(
    config,
    ckpt_path=None,
    devices=None,
    eval_freq=None,
    max_steps=-1,
    max_epochs=None,
    accumulate_grad_batches=1,
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    run_name="",
    precision="bf16-mixed",
    accelerator="gpu",
    dryrun=False,
    **kwargs,
) -> pl.Trainer:

    wandb_logger = (
        WandbLogger(
            project=wandb_project_name,
            entity=wandb_entity,
            name=run_name,
            config=config,
        )
        if not dryrun
        else None
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_bleu",
        dirpath=ckpt_path,
        filename="{epoch:02d}-{val_bleu:.2f}",
        save_last=True,
        save_top_k=2,
        mode="max",
    )

    return pl.Trainer(
        logger=wandb_logger,  # FIXME comment for testing
        devices=devices,
        accelerator=accelerator,
        max_steps=max_steps,
        max_epochs=max_epochs,
        val_check_interval=eval_freq,
        log_every_n_steps=log_every_n_steps,
        callbacks=[model_checkpoint],
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
    )


def get_config(model_cls, model_config):
    base = {
        "train_batch_size": 64,
        "val_batch_size": 64,
        "accumulate_grad_batches": 1,
        "gradient_clip_val": None,
        "max_steps": 1000000,
        "learning_rate": 1e-3,
        "dropout": 0.3,
        "weight_decay": 0.001,
        "warmup_steps": 0, # FIXME
        "eval_freq": 500,
        "log_every_n_steps": 5,
        "devices": None,
        "resume_path": None,
    }

    assert (
        model_config in model_cls.configs
    ), f"config {model_config} not found for {model.model_name}"
    specific = model_cls.configs[model_config]

    #  merges dicts
    return specific | base


if __name__ == "__main__":
    args, _ = parser.parse_known_args()

    model_name = args.model_name
    model_config = args.model_config
    ds_name = args.dataset
    source, target = args.language_pair

    model_cls = build_model(task="mt", name=model_name)
    logging.info(f"building dataset {ds_name} for {source}-{target}")

    dataset = build_dataset(
        ds_name,
        source,
        target,
        is_encoder_decoder=model_cls.is_encoder_decoder,
    )
    tokenizer = dataset.get_tokenizer()

    if model_cls.model_name == "mamba2":
        tokenizer.padding_side = "right"

    config = get_config(model_cls=model_cls, model_config=model_config)
    config["ckpt_path"] = dataset.get_ckpt_path(model_name)
    config["vocab_size"] = tokenizer.vocab_size
    config = config | vars(args)

    logging.info(f"config: {config}")
    train_dl, val_dl, _ = dataset.get_dataloaders(
        tokenizer,
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
    )

    run_name_suffix = args.run_name_suffix
    trainer = get_trainer(
        config=config,
        run_name=f"{dataset.name} {source}-{target} {model_name} {run_name_suffix}",
        **config,
    )

    logging.info("building model")
    model = model_cls(config=config, tokenizer=tokenizer, **config)
    resume_training_path = config["resume_path"]  # defaults to None

    if config["dryrun"]:
        logging.info("dryrun, raw training")
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_training_path)

    else:
        max_retries = 10
        while max_retries > 0:
            try:
                trainer.fit(model, train_dl, val_dl, ckpt_path=resume_training_path)
                break
            except torch.cuda.OutOfMemoryError:
                logging.info("OOM, retrying...")
                torch.cuda.empty_cache()
                gc.collect()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"unknown exception: \n{e}")
            finally:
                max_retries -= 1
                resume_training_path = "last"

        trainer.save_checkpoint(
            os.path.join(config["ckpt_path"], f"final-{trainer.logger.version}.ckpt")
        )

import os
import logging
import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils.mt.argparser import parser


from models import build_model
from mt.ds import build_dataset

logging.basicConfig(level=logging.INFO)

wandb_project_name = ""
wandb_entity = ""


def get_trainer(
    config,
    ckpt_path=None,
    devices=None,
    eval_freq=None,
    max_steps=-1,
    max_epochs=None,
    log_every_n_steps=10,
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

    return pl.Trainer(
        logger=wandb_logger,
        devices=devices,
        accelerator=accelerator,
        max_steps=max_steps,
        max_epochs=max_epochs,
        val_check_interval=eval_freq,
        log_every_n_steps=log_every_n_steps,
        precision=precision,
    )


def get_config(model_cls, model_config):
    base = {
        "test_batch_size": 16,
        "test": True,
        "log_every_n_steps": 5,
        "devices": None,
        "resume_path": None,
        "dropout": 0,  # would be disabled anyway
    }

    assert (
        model_config in model_cls.configs
    ), f"config {model_config} not found for {model.model_name}"
    specific = model_cls.configs[model_config]

    #  merges dicts
    return base | specific


if __name__ == "__main__":
    args = parser.parse_args()

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
    # FIXME build if not found? currently throws assertion error
    tokenizer = dataset.get_tokenizer()

    config = get_config(model_cls=model_cls, model_config=model_config)
    config["ckpt_path"] = dataset.get_ckpt_path(model_name)
    config["vocab_size"] = tokenizer.vocab_size
    config = config | vars(args)

    _, _, test_dl = dataset.get_dataloaders(
        tokenizer,
        test_batch_size=config["test_batch_size"],
        test=True,
    )
    trainer = get_trainer(
        config=config,
        run_name=f"{dataset.name} {source}-{target} {model_name}",
        **config,
    )

    logging.info("building model")
    model_cls = build_model(task="mt", name=model_name)
    model = model_cls(config=config, tokenizer=tokenizer, **config)
    model.load_state_dict(
        torch.load(
            config["resume_path"],
            map_location=torch.device("cpu"),
        )["state_dict"]
    )

    res = trainer.test(model=model, dataloaders=test_dl)

    logging.info(f"results: {res}")

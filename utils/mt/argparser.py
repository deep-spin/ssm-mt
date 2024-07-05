import argparse
from typing import Sequence

parser = argparse.ArgumentParser(
    description="Training script for machine translation models."
)

parser.add_argument(
    "--model_name", type=str, required=True, help="Name of the model to be used."
)
parser.add_argument(
    "--devices",
    type=int,
    nargs="+",
    help="List of device IDs to use for training.",
)
parser.add_argument(
    "--resume_path",
    type=str,
    default=None,
    help="Path to a checkpoint file to resume training.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="Dataset to use for training.",
)
parser.add_argument(
    "--language_pair",
    nargs=2,
    type=str,
    help="Language pair to use for training.",
)

parser.add_argument(
    "--use_padding",
    action="store_true",
    default=False,
    help="Whether to use padding for the model.",
)

parser.add_argument(
    "--model_config",
    default="default",
    type=str,
    help="Configuration to use for the model.",
)

parser.add_argument(
    "--precision",
    default="bf16-mixed",
    type=str,
    help="Precision to use for training.",
)

parser.add_argument(
    "--dryrun",
    action="store_true",
    default=False,
    help="Whether to do a dryrun.",
)

parser.add_argument(
    "--test_per_sample",
    action="store_true",
    default=False,
    help="Whether to test per sample.",
)

parser.add_argument(
    "--test_suffix",
    type=str,
    default="",
    help="Suffix to add to the test results.",
)

parser.add_argument(
    "--run_name_suffix",
    type=str,
    default="",
    help="Name of the run.",
)

parser.add_argument(
    "--layers_config_key",
    type=str,
    default="interleaved",
    help="Key to use for layers configuration for hybrids.",
)


parser.add_argument(
    "--window_size",
    type=int,
    default=64,
    help="Window size used for sliding window attention.",
)

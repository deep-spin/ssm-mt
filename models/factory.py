from models.retnet import RetNetClassification, RetNetMT

# from models.gateloop import GateLoopMT
from models.transformer import TransformerMT, TransformerEncDecMT, LlamaMT
from models.mamba import MambaMT
from models.hybrids.tr_mamba import HybridTrMamba
from models.hybrids.zero_mamba import HybridAttMamba
from models.hybrids.local_mamba import HybridLocalMamba
from models.hybrids.mamba_mistral import HybridMistralMamba

from models.hybrids.sliding_mamba import HybridSlidingMamba
from models.hybrids.mamba_encdec import MambaEncDec
from models.mamba2.mt_wrapper import Mamba2MT


# from models.hybrids.based_mamba import BasedMambaMT
from pytorch_lightning import LightningModule


def _cls_model(name):
    model_dict = {
        "retnet": RetNetClassification,
    }

    return model_dict.get(name)


def _mt_model(name):
    model_dict = dict(
        (cls.model_name, cls)
        for cls in (
            RetNetMT,
            TransformerMT,
            TransformerEncDecMT,
            LlamaMT,
            MambaMT,
            HybridTrMamba,
            HybridAttMamba,
            HybridLocalMamba,
            HybridSlidingMamba,
            HybridMistralMamba,
            MambaEncDec,
            Mamba2MT
        )
    )

    assert name in model_dict.keys(), f"model {name} not found"
    return model_dict.get(name)


def _task_select(task, name):
    task_dict = {
        "mt": _mt_model,
        "cls": _cls_model,
    }

    return task_dict.get(task)(name)


def build_model(task, name) -> LightningModule:
    return _task_select(task, name)


def load_ckpt(model_cls: LightningModule, path: str, **kwargs):
    return model_cls.load_from_checkpoint(path, **kwargs)

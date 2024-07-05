from mt.ds.base import BaseDataset
from mt.ds.iwslt17 import IWSLT17Dataset
from mt.ds.kftt import KFTTDataset
from mt.ds.news import NewsDataset
from mt.ds.wmt14 import WMT14Dataset
from mt.ds.wmt16 import WMT16Dataset
from mt.ds.wmt16_processed import WMT16Dataset as WMT16ProcessedDataset
from mt.ds.wmt23 import WMT23Dataset
from mt.ds.wmt236M import WMT23Dataset as WMT23Dataset6M
from mt.ds.wmt23concat5 import WMT23Dataset as WMT23Concat5
from mt.ds.wmt23concat10 import WMT23Dataset as WMT23Concat10
from mt.ds.wmt23_ende_test import WMT23Dataset as WMT23EnDeTest
from mt.ds.ted_talks_val import TedTalksDataset


def build_dataset(
    name: str,
    source: str,
    target: str,
    is_encoder_decoder: bool,
) -> BaseDataset:

    ds_dict = dict(
        (cls.name, cls)
        for cls in (
            IWSLT17Dataset,
            WMT16Dataset,
            KFTTDataset,
            WMT14Dataset,
            NewsDataset,
            WMT23Dataset,
            WMT16ProcessedDataset,
            WMT23Dataset6M,
            WMT23Concat5,
            WMT23Concat10,
            WMT23EnDeTest,
            TedTalksDataset,
        )
    )

    assert name in ds_dict.keys(), f"Dataset {name} not supported."
    return ds_dict[name](source, target, is_encoder_decoder)

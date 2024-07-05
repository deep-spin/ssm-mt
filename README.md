# How Effective are State Space Models for Machine Translation?

PyTorch implementation of the experiments described in our paper "How Effective are State Space Models for Machine Translation?" [1]. This repository contains code to replicate the results reported in the paper, focusing on the comparison between State Space Models (SSMs) and traditional Transformer architectures for machine translation tasks.


## Finetuning Experiments

For experiments involving finetuning of pretrained models, please refer to our separate repository: [SSM-MT-Finetuning](https://github.com/yourusername/ssm-mt-finetuning)


## Requirements

To set up the environment for our experiments, follow these steps:


- Setup local `mamba_ssm` package as defined in our [fork](https://github.com/xtwigs/mamba).
- Install required packages in `requirements.txt`.
- Some datasets are obtained from huggingface others are local files. You can check the init function in `mt/ds/{dataset}.py` to see how to obtain the data.


## Training

To train the models described in the paper, use the following command:

```
python train.py --model mamba --dataset wmt14 --langauge_pair en de --devices 0 --use_padding
```

Model and dataset names and dataset can be found in `models/factory.py` and `mt/ds/factory.py`, respectively.
Additonal configuration is set in `mt/run.py` and through command line arguments defined in `utils/mt/argparser.py`.



## References

[1] Hugo Pitorro*, Pavlo Vasylenko*, Marcos Treviso, André F. T. Martins. "How Effective are State Space Models for Machine Translation?" Submitted to EMNLP 2024.

\* Equal contribution

## Citation

If you use this code or our results in your research, please cite our paper:

```
@article{pitorro2024effective,
  title={How Effective are State Space Models for Machine Translation?},
  author={Pitorro, Hugo and Vasylenko, Pavlo and Treviso, Marcos and Martins, André F. T.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
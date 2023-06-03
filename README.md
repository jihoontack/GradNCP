# GradNCP

Official PyTorch implementation of **["Learning Large-scale Neural Fields via Context Pruned Meta-Learning"](https://arxiv.org/abs/2302.00617)** by
[Jihoon Tack](https://jihoontack.github.io/),
[Subin Kim](https://subin-kim-cv.github.io/), 
[Sihyun Yu](https://sihyun.me/), 
[Jaeho Lee](https://jaeho-lee.github.io/), 
[Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html), 
[Jonathan Richard Schwarz](https://jonathan-schwarz.github.io/).
\
Currently, only CelebA is supported (will be updated soon).

**TL;DR**: *We propose an efficient meta-learning framework for scalable neural fields learning that involves online data pruning of the context set.*
<p align="center">
    <img src=figure/concept_figure.png width="900"> 
</p>

## 1. Dependencies
```bash
conda create -n gradncp python=3.8 -y
conda activate gradncp

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install einops pyyaml tensorboardX tensorboard pytorch_msssim lpips
```

## 2. Dataset
- Dataset path `/data`
- Download CelebA

## 3. How to run?
### Train
```bash
# Learnit
CUDA_VISIBLE_DEVICES=0 python main.py --configs ./configs/main/maml_celeba.yaml

# Ours
CUDA_VISIBLE_DEVICES=0 python main.py --configs ./configs/main/ours_celeba.yaml
```

### Evaluation
- Example of `<PATH TO CHECKPOINT>`: `./logs/maml_celeba/best.pth`
```bash
# Learnit
CUDA_VISIBLE_DEVICES=0 python eval.py --configs ./configs/evaluation/eval_celeba.yaml --load_path ./logs/xxxx/best.model

# Ours (CelebaA) Example
CUDA_VISIBLE_DEVICES=0 python eval.py --configs ./configs/evaluation/eval_celeba_ours.yaml --load_path ./logs/xxxx/best.model
```

## Reference
- [JAX Learnit](https://github.com/tancik/learnit)
- [PyTorch Siren](https://github.com/lucidrains/siren-pytorch)
- [PyTorch MetaSDF](https://github.com/vsitzmann/metasdf)
- [PyTorch Meta-SparseINR](https://github.com/jaeho-lee/MetaSparseINR)
- [PyTorch COIN++](https://github.com/EmilienDupont/coinpp)

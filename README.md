
## 

This repository is a reimplementation of zero-shot control methods described in the following papers by Hazarika et al.:

- Attention Biasing and Context Augmentation for Zero-Shot Control of Encoder-Decoder Transformers for Natural Language Generation (2022)
- Zero-Shot Controlled Generation with Encoder-Decoder Transformers (2021/2022)



## Setup

```
conda create -n unsup_ctrl python=3.8
conda activate unsup_ctrl
pip install -r requirements.txt
```

### Data

Experiments in the original paper mostly use the Topical Chat dataset ([https://m.media-amazon.com/images/G/01/amazon.jobs/3079_Paper._CB1565131710_.pdf](Gopalakrishnan_et_al_2019)). To get this, see the instructions at [https://github.com/alexa/Topical-Chat].
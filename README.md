
# Can Wikipedia Help Offline RL? 

Machel Reid, Yutaro Yamada and Shixiang Shane Gu.

Our paper is up on [arXiv](https://arxiv.org/abs/2201.12122).

## Overview

Official codebase for [Can Wikipedia Help Offline Reinforcement Learning?](https://arxiv.org/abs/2201.12122).
Contains scripts to reproduce experiments. (This codebase is based on that of https://github.com/kzl/decision-transformer)

![image info](./architecture.png)

## Instructions

We provide code our `code` directory containing code for our experiments.
### Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

### Downloading datasets

Datasets are stored in the `data` directory. LM co-training and vision experiments can be found in `lm_cotraining` and `vision` directories respectively.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

### Downloading ChibiT

ChibiT can be downloaded with gdown as follows:
```bash
gdown --id $ID #we will add it soon!
```

### Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env hopper --dataset medium --model_type dt --pretrained_lm gpt2 \ # or path to chibiT
--gpt_kmeans --gpt_kmeans-const 0.1 
--
```

The `run.sh` file has example commands.

Adding `-w True` will log results to Weights and Biases.

## Citation

Please cite our paper as:

```
@misc{reid2022wikipedia,
      title={Can Wikipedia Help Offline Reinforcement Learning?}, 
      author={Machel Reid and Yutaro Yamada and Shixiang Shane Gu},
      year={2022},
      eprint={2201.12122},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

MIT

# 🎬 Vid2Sim 🤖: Realistic and Interactive Simulation from Video for Urban Navigation
> [Ziyang Xie](https://ziyangxie.site/), [Zhizheng Liu](https://scholar.google.com/citations?user=Asc7j9oAAAAJ&hl=en), [Zhenghao Peng](https://pengzhenghao.github.io/), [Wayne Wu](https://wywu.github.io/), [Bolei Zhou](https://boleizhou.github.io/)
>
> [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2501.06693)
> [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://metadriverse.github.io/vid2sim/)

Vid2Sim is a novel framework that converts monocular videos into photorealistic and physically interactive simulation environments for training embodied agents with minimal sim-to-real gap.

<p align="center">
  <img src="../../assets/teaser.png" width="100%">
</p>

> [!Note]
> This subfolder is used for agent RL training in the reconstructed simulation environment.  
> Please follow the instructions below to build the environment and run the RL training.  
>     
> *For simulation environment reconstruction, please refer to the [Vid2Sim-Recon](https://github.com/Vid2Sim/Vid2Sim/tree/main/src/vid2sim_recon) subfolder.*


## 🔧 1. Installation
Follow the instructions in `src/vid2sim_rl/requirements.txt` to install the required packages.
```bash
pip install -r requirements.txt
```

### 📦 Code Structure
```
vid2sim_rl/
├── config/ # Training configuration
├── envs/ # Unity environments
├── unity_envs/ # Unity environment scripts (C#)
├── train.py # Training script
├── infer.py # Inference script
├── requirements.txt # Requirements
├── ...
```

## 🏗️ 2. RL Environment Setup (Unity Version)
Vid2Sim use Unity [ml-agents](https://github.com/Unity-Technologies/ml-agents) to convert compiled Unity environment into commonly used OpenAI [gym](https://github.com/Farama-Foundation/Gymnasium) environment to support RL training with [stable-baseline3](https://github.com/DLR-RM/stable-baselines3). For the specific implementation, you can refer to the `UnityEnvWrapper` in [env.py](env.py) file for more details.

To get the training environment, you can use the provided unity environments in [🗂️ Vid2Sim dataset](https://drive.google.com/drive/folders/1LCruqb6M3mCgsjaqI1ON6WVoZ-9CmQDY?usp=sharing) or you could build your own Unity environments.
In [BUILD_YOUR_OWN_ENV.md](BUILD_YOUR_OWN_ENV.md), we provide the instructions to build your own envs from scratch.

After you get the compiled environment executable, you can set the `env_paths` parameter in the training configuration to the path of the environment executable.
```yaml
env:
  env_paths: [
    "path/to/your/environment/executable"
  ]
```

## 🤖 3. Training Configuration

The training configuration is stored in the `config` folder. You can modify the configuration to suit your needs. Most of the config terms are self-explanatory.
```yaml
# Agent visual observation resolution
obs_width: 128
obs_height: 72

# Reward Settings
max_episode_length: 60
time_penalty_multiplier: -0.1
distance_reward_multiplier: 1
steer_smooth_reward_multiplier: 0.05
collision_penalty: -1
goal_reward: 10
fail_reward: -10
collision_limit: 5

# Obstacle Settings
random_obj_placement: True
max_loaded_obj_num: 1

# Social Navigation Settings
enable_dynamic_agent: False
dynamic_agent_n: 1

# Agent Observation Settings
use_rgb: True
use_depth: False
```


## 🚀 4. Train Agent in Real2Sim Environments
To train the agent in the Real2Sim environments, you can use the following command:

```bash
python train.py --config <PATH_TO_CONFIG>
```

If you are training on a remote server with no graphic display, we recommend using [Xvfb](https://en.wikipedia.org/wiki/Xvfb) to run the training.
```bash
xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24' python train.py --config_name <CONFIG_NAME>
```

## 📊 5. Evaluation
To evaluate the trained agent, you can use the following command:

```bash
python infer.py --config <PATH_TO_CONFIG>
```

or on remote server:

```bash
xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24' python infer.py --config_name <CONFIG_NAME>
```

[Recommend] Using the script to start GPU training
```bash
./train.sh
```
ensure the `env_class: 'subproc'` to start environments in subprocess concurrently.


## Citation 📝

If you find this work useful in your research, please consider citing:

```bibtex
@article{xie2024vid2sim,
  title={Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation},
  author={Xie, Ziyang and Liu, Zhizheng and Peng, Zhenghao and Wu, Wayne and Zhou, Bolei},
  journal={CVPR},
  year={2025}
}
```

# VecPyPnC

Vectorized PyTorch implementation of the [PyPnC](https://github.com/junhyeokahn/PyPnC) planning and control
stack for the Draco3 humanoid, enabling batched simulation for reinforcement learning on top of a
model-based controller.

## Overview

| Path | Contents |
|---|---|
| `pnc_pytorch/planner/locomotion` | ALIP footstep MPC and swing-foot trajectories |
| `pnc_pytorch/wbc/ihwbc` | Implicit-hierarchy whole-body control |
| `pnc_pytorch/draco3_pnc` | Draco3 control architecture and state machine |
| `simulator/pybullet/rl`, `rl_model/` | RL environment and PPO training |

The original PyPnC code is kept alongside.

## Usage

```sh
conda env create -f pypnc.yml && conda activate pypnc_pytorch
pip install gymnasium stable-baselines3      # for RL only

python simulator/pybullet/draco3_pytorch.py  # batched controller in PyBullet
python simulator/pybullet/rl/train.py        # PPO training
```

## Credits

Based on PyPnC by Junhyeok Ahn et al.:

```bibtex
@article{ahn2021versatile,
  title={Versatile Locomotion Planning and Control for Humanoid Robots},
  author={Ahn, Junhyeok and Jorgensen, Steven and Bang, Seung Hyeon and Sentis, Luis},
  journal={Frontiers in Robotics and AI},
  year={2021}
}
```

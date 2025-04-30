# rl-policy-gradient-benchmark

This repository implements and compares **REINFORCE** and **Proximal Policy Optimization (PPO)** algorithms on classic continuous control environments using Gymnasium.

---

## 📦 Python Environment Installation Instructions

Make sure you have **conda** installed on your system.  
Follow this guide: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation

Then, from the root of this repository:

    conda env create -f conda_env.yml
    conda activate ddrl_a4

---

## 🚀 Running the Code

Use the following command format to train either REINFORCE or PPO:

    python run.py --alg reinforce --seed 42 --env <env_name>
    python run.py --alg ppo --seed 42 --env <env_name>

Replace <env_name> with any of the supported Gymnasium environments, such as:

- Pendulum-v1
- BipedalWalker-v3
- LunarLanderContinuous-v2

Example:

    python run.py --alg reinforce --seed 42 --env Pendulum-v1

---

## 📊 Features

- Implements REINFORCE and PPO from scratch using PyTorch
- Uses structured logging and CSV-based reward tracking
- Records 10 evaluation videos evenly spaced over training
- Automatically saves trained models and metrics
- Supports configurable seed and environment selection

---

## 📁 Project Structure

    .
    ├── run.py                  # Main entry point for training
    ├── reinforce.py            # REINFORCE training logic
    ├── ppo.py                  # PPO training logic
    ├── base_alg.py             # Shared training logic and rollout code
    ├── video.py                # Evaluation video recorder
    ├── conda_env.yml           # Conda environment file
    ├── results/                # Output directory for logs, videos, CSVs
    └── README.md

---

## 📈 Output

- Evaluation Videos: saved every ~10% of training to  
  results/eval_video/reinforce_eval_stepXXXX.mp4

- Reward Logs: CSV files saved to  
  results/<date>/reinforce_rewards_seed<seed>\_env<env>.csv

- Model Checkpoints: actor weights saved every save_freq steps

---

## 🧠 Notes

- Training may take several minutes depending on the environment.
- To save videos as .mp4, ensure ffmpeg is installed via:
  conda install -c conda-forge ffmpeg
  or:
  brew install ffmpeg

- Use different --seed values for reproducibility experiments.

---

## ✏️ Author

This code was originally developed by **Eric Yu** as part of _CSCI-GA 3033: Deep Decision Making_ at NYU, taught by **Lerrel Pinto**.  
It has been modified and extended based on the original implementation.

"""
    This file will run REINFORCE or PPO code
    with the input seed and environment.
"""

import gymnasium as gym
import os
import argparse

# Import ppo files
from ppo.ppo import PPO
from ppo.reinforce import REINFORCE
from ppo.network import FeedForwardNN


def train_ppo(args):
    """
        Trains with PPO on specified environment.

        Parameters:
            args - the arguments defined in main.

        Return:
            None
    """
    # Store hyperparameters and total timesteps to run by environment
    hyperparameters = {}
    total_timesteps = 0
    if args.env == 'Pendulum-v1':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 10,
                            'lr': 3e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 2005000
    elif args.env == 'BipedalWalker-v3':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 1600, 'gamma': 0.99, 'n_updates_per_iteration': 10,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6405000
    elif args.env == 'LunarLanderContinuous-v2':
        hyperparameters = {'timesteps_per_batch': 1024, 'max_timesteps_per_episode': 1000, 'gamma': 0.999, 'n_updates_per_iteration': 4,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6005000
    else:
        raise ValueError("Unrecognized environment, please specify the hyperparameters first.")

    # Make the environment and model, and train
    env = gym.make(args.env)
    model = PPO(FeedForwardNN, env, **hyperparameters)
    model.learn(total_timesteps)


def train_reinforce(args):
    """
        Trains with REINFORCE on specified environment.

        Parameters:
            args - the arguments defined in main.

        Return:
            None
    """
    # Store hyperparameters and total timesteps to run by environment
    hyperparameters = {}
    total_timesteps = 0
    if args.env == 'Pendulum-v1':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 200, 'gamma': 0.99, 'n_updates_per_iteration': 1,
                            'lr': 3e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 2005000
    elif args.env == 'BipedalWalker-v3':
        hyperparameters = {'timesteps_per_batch': 2048, 'max_timesteps_per_episode': 1600, 'gamma': 0.99, 'n_updates_per_iteration': 1,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6405000
    elif args.env == 'LunarLanderContinuous-v2':
        hyperparameters = {'timesteps_per_batch': 1024, 'max_timesteps_per_episode': 1000, 'gamma': 0.999, 'n_updates_per_iteration': 1,
                            'lr': 2.5e-4, 'clip': 0.2, 'save_freq': 1e6, 'seed': args.seed}
        total_timesteps = 6005000
    else:
        raise ValueError("Unrecognized environment, please specify the hyperparameters first.")

    # Make the environment and model, and train
    env = gym.make(args.env)
    model = REINFORCE(FeedForwardNN, env, **hyperparameters)
    model.learn(total_timesteps)


def main(args):
    """
        An intermediate function that will call either REINFORCE learn or PPO learn.

        Parameters:
            args - the arguments defined below

        Return:
            None
    """
    if args.alg == 'PPO':
        train_ppo(args)
    elif args.alg == 'reinforce':
        train_reinforce(args)
    else:
        raise ValueError(f'Algorithm {args.alg} not defined; options are reinforce or PPO.')

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', dest='alg', type=str, default='reinforce')        # Formal name of our algorithm
    parser.add_argument('--seed', dest='seed', type=int, default=None)             # An int for our seed
    parser.add_argument('--env', dest='env', type=str, default='')                 # Formal name of environment

    args = parser.parse_args()

    main(args)
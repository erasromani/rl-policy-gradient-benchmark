"""
The file contains the REINFORCE class to train with.
"""

import csv
import os

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from .base_alg import BasePolicyGradient
from .video import VideoRecorder


class REINFORCE(BasePolicyGradient):
    """
    This is the REINFORCE class we will use as our model in run.py
    """

    def learn(self, total_timesteps):
        """
        Train the actor network. Here is where the main REINFORCE algorithm resides.

        Parameters:
            total_timesteps - the total number of timesteps to train for

        Return:
            None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        self.training_rewards = []
        video_record_interval = total_timesteps // 10
        next_video_step = video_record_interval
        video_recorder = VideoRecorder(root_dir=self.output_dir)
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )
            batch_rewards = [np.sum(ep_rews) for ep_rews in self.batch_rews]
            self.training_rewards.extend(batch_rewards)

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # This is the loop where we update our network for some n epochs
            # NOTE: This n_updates_per_iteration should always be 1 for REINFORCE.
            assert self.n_updates_per_iteration == 1
            for _ in range(self.n_updates_per_iteration):
                # Calculate actor loss.
                # NOTE: we take the negative loss because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                reward_to_go, log_probs = self.evaluate(
                    batch_obs, batch_acts, batch_rtgs
                )
                normalized_rewards = (reward_to_go - reward_to_go.mean()) / (
                    reward_to_go.std() + 1e-10
                )
                actor_loss = (-normalized_rewards * log_probs).mean()
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Log actor loss
                self.logger["actor_losses"].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            if t_so_far >= next_video_step:
                eval_env = gym.make(self.env.spec.id, render_mode="rgb_array")
                obs, _ = eval_env.reset()
                done = False
                video_recorder.init(eval_env, enabled=True)

                for _ in range(self.max_timesteps_per_episode):
                    obs_tensor = (
                        torch.tensor(obs, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    with torch.no_grad():
                        action = (
                            self.actor(obs_tensor).squeeze(0).cpu().numpy()
                        )

                    obs, _, done, truncated, _ = eval_env.step(action)
                    video_recorder.record(eval_env)

                    if done or truncated:
                        break

                video_recorder.save(f"reinforce_eval_iter{i_so_far}.mp4")
                eval_env.close()

                next_video_step += video_record_interval

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./REINFORCE_actor.pth")

        os.makedirs(self.output_dir, exist_ok=True)
        filename = (
            f"reinforce_rewards_seed{self.seed}_env{self.env.spec.id}.csv"
        )
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])
            for idx, r in enumerate(self.training_rewards):
                writer.writerow([idx, r])

        print(f"[INFO] Saved reward log to {filepath}")

    def evaluate(self, batch_obs, batch_acts, batch_rtgs):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
            batch_obs - the observations from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of observation)
            batch_acts - the actions from the most recently collected batch as a tensor.
                        Shape: (number of timesteps in batch, dimension of action)
            batch_rtgs - the rewards-to-go calculated in the most recently collected
                            batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return batch_rtgs, log_probs

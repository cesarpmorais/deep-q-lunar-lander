import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes, CallbackList
from dqn import plot_training_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals['rewards'][0]
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    env = TimeLimit(env, max_episode_steps=1000)

    model = DQN(
        "MlpPolicy", env, 
        learning_rate=5e-4,
        gamma=0.99, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.005,
        buffer_size=100000,
        batch_size=64,
        target_update_interval=10,
        verbose=1, 
        device=device)
    reward_callback = RewardLogger()
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=1000, verbose=1)
    callback = CallbackList([reward_callback, stop_callback])

    model.learn(total_timesteps=1000000, callback=callback)
    model.save("dqn_lunar_lander_baseline")

    rewards = reward_callback.episode_rewards

    plot_training_results(rewards, [], 1)

    env.close()
    
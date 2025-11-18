import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import json
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

def calculateBaseline(learning_rate=5e-4,
        gamma=0.99, 
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.005,
        buffer_size=100000,
        batch_size=64,
        target_update_interval=10):
    
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
        verbose=0, 
        device=device)
    reward_callback = RewardLogger()
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=1000, verbose=0)
    callback = CallbackList([reward_callback, stop_callback])

    model.learn(total_timesteps=1000000, callback=callback)
    model.save("dqn_lunar_lander_baseline")

    rewards = reward_callback.episode_rewards
    mean_reward = np.mean(rewards)
    print(f"Recompensa media por episodio: {mean_reward}")

    plot_training_results(rewards, [], 1)

    env.close()

    return mean_reward

if __name__ == "__main__":
    top5 = []

    for i in range(5):
        with open("top5_configs", "r") as f:
            data = json.load(f)
            top5.append(data[i]["config"])
        
        print(f"\nCalculando baseline para a configuração top-{i+1}: {top5[i]}")
        baseline_avg = calculateBaseline(
            learning_rate=top5[i]["learning_rate"],
            gamma=top5[i]["gamma"],
            exploration_initial_eps=top5[i]["epsilon_start"],
            exploration_final_eps=top5[i]["epsilon_end"],
            exploration_fraction=top5[i]["epsilon_decay"],
            buffer_size=top5[i]["buffer_capacity"],
            batch_size=top5[i]["batch_size"],
            target_update_interval=top5[i]["target_update_freq"],
        )
        print(f"Recompensa média baseline: {baseline_avg}\n")
        print(f"Recompensa média do top-{i+1}: {data[i]['mean_reward']}\n")
    



    
    
    calculateBaseline()
    
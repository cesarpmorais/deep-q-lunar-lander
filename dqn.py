import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

# Use GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]
    ):
        super(DQNetwork, self).__init__()

        layers = []
        input_dim = state_dim

        # hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dims: List[int] = [128, 128],
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Main and target networks
        self.q_network = DQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = DQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Adam optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.loss_history = []
        self.epsilon_history = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Calcuates target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training - is it necessary?
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Updates target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.loss_history.append(loss.item())
        self.epsilon_history.append(self.epsilon)

        return loss.item()

    def save(self, filepath: str):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )
        print(f"Modelo salvo em: {filepath}")

    def load(self, filepath: str):
        """Carrega o modelo"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        print(f"Modelo carregado de: {filepath}")


def train_dqn(
    agent: DQNAgent,
    env: gym.Env,
    num_episodes: int = 1000,
    max_steps: int = 1000,
    eval_freq: int = 50,
    eval_episodes: int = 10,
    save_model: bool = True,
    save_path: str = "dqn_lunar_lander.pth",
):
    episode_rewards = []
    eval_rewards = []
    best_eval_reward = -float("inf")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            loss = agent.train_step()

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, eval_episodes)
            eval_rewards.append(eval_reward)

            print(f"Episódio {episode + 1}/{num_episodes}")
            print(
                f"  Recompensa Média (Treino): {np.mean(episode_rewards[-eval_freq:]):.2f}"
            )
            print(f"  Recompensa Média (Eval): {eval_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")

            if save_model and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(save_path)

    return episode_rewards, eval_rewards


def evaluate_agent(agent: DQNAgent, env: gym.Env, num_episodes: int = 10) -> float:
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def plot_training_results(
    episode_rewards: List[float], eval_rewards: List[float], eval_freq: int
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(episode_rewards, alpha=0.3, label="Episódio")
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        axes[0].plot(
            range(window - 1, len(episode_rewards)),
            moving_avg,
            label=f"Média Móvel ({window})",
        )
    axes[0].set_xlabel("Episódio")
    axes[0].set_ylabel("Recompensa")
    axes[0].set_title("Recompensas de Treinamento")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    eval_episodes = [i * eval_freq for i in range(1, len(eval_rewards) + 1)]
    axes[1].plot(eval_episodes, eval_rewards, marker="o", label="Avaliação")
    axes[1].set_xlabel("Episódio")
    axes[1].set_ylabel("Recompensa Média")
    axes[1].set_title("Recompensas de Avaliação")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")

    config = {
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_capacity": 100000,
        "batch_size": 64,
        "target_update_freq": 10,
        "hidden_dims": [128, 128],
    }

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        **config,
    )

    print("Iniciando treinamento...")
    episode_rewards, eval_rewards = train_dqn(
        agent=agent,
        env=env,
        num_episodes=1000,
        max_steps=1000,
        eval_freq=100,
        eval_episodes=10,
        save_model=True,
        save_path="dqn_lunar_lander_best.pth",
    )

    plot_training_results(episode_rewards, eval_rewards, eval_freq=50)

    print("\nAvaliação final...")
    final_reward = evaluate_agent(agent, env, num_episodes=100)
    print(f"Recompensa média (100 episódios): {final_reward:.2f}")

    env.close()

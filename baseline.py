import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    StopTrainingOnMaxEpisodes,
    CallbackList,
)
from dqn import plot_training_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True


def calculateBaseline(
    learning_rate=5e-4,
    gamma=0.99,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.005,
    buffer_size=100000,
    batch_size=64,
    target_update_interval=10,
    run_id=0,
):

    env = gym.make("LunarLander-v3")
    env = TimeLimit(env, max_episode_steps=1000)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_interval=target_update_interval,
        verbose=0,
        device=device,
    )
    reward_callback = RewardLogger()
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=1000, verbose=0)
    callback = CallbackList([reward_callback, stop_callback])

    model.learn(total_timesteps=1000000, callback=callback, progress_bar=True)
    model.save(f"dqn_lunar_lander_baseline_run_{run_id}")

    rewards = reward_callback.episode_rewards
    mean_reward = np.mean(rewards[-50:])
    print(f"Run {run_id}: Recompensa media por episodio: {mean_reward}")

    env.close()

    return mean_reward


if __name__ == "__main__":
    # Load the top 5 configurations
    with open("top5_configs.json", "r") as f:
        data = json.load(f)

    top5_configs = [entry["config"] for entry in data]
    num_runs = 5  # Match the hyperparameter pipeline Phase B

    baseline_results = []

    # Calculate baseline for each configuration
    for i in range(5):
        config = top5_configs[i]
        print(f"\n{'='*60}")
        print(f"Calculando baseline para a configuração top-{i+1}")
        print(f"Configuração: {config}")
        print(f"{'='*60}")

        # Run multiple times for this configuration
        run_rewards = []
        for run in range(num_runs):
            print(f"\nExecutando run {run+1}/{num_runs}")
            reward = calculateBaseline(
                learning_rate=config["learning_rate"],
                gamma=config["gamma"],
                exploration_initial_eps=config["epsilon_start"],
                exploration_final_eps=config["epsilon_end"],
                exploration_fraction=1 - config["epsilon_decay"],
                buffer_size=config["buffer_capacity"],
                batch_size=config["batch_size"],
                target_update_interval=config["target_update_freq"],
                run_id=run,
            )
            run_rewards.append(reward)

        # Calculate statistics
        mean_reward = np.mean(run_rewards)
        std_reward = np.std(run_rewards)

        baseline_results.append(
            {
                "config_index": i,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "run_rewards": [float(r) for r in run_rewards],
            }
        )

        print(f"\nEstatísticas para configuração top-{i+1}:")
        print(f"Recompensa média: {mean_reward:.2f}")
        print(f"Desvio padrão: {std_reward:.2f}")
        print(f"Recompensas individuais: {[f'{r:.2f}' for r in run_rewards]}")

    # Write detailed comparison results
    print(f"\n{'='*80}")
    print("COMPARAÇÃO FINAL - BASELINE vs HYPERPARAMETER SEARCH")
    print(f"{'='*80}")

    with open("baseline_comparison_detailed.txt", "w") as f:
        f.write("COMPARAÇÃO DETALHADA - BASELINE vs HYPERPARAMETER SEARCH\n")
        f.write("=" * 60 + "\n\n")

        for i in range(5):
            baseline_result = baseline_results[i]
            original_result = data[i]

            difference = original_result["mean_reward"] - baseline_result["mean_reward"]

            print(f"Configuração top-{i+1}:")
            print(
                f"  Baseline - Média: {baseline_result['mean_reward']:.2f} ± {baseline_result['std_reward']:.2f}"
            )
            print(
                f"  Original - Média: {original_result['mean_reward']:.2f} ± {original_result['std_reward']:.2f}"
            )
            print(f"  Diferença: {difference:.2f}")
            print(f"  Baseline superior: {'Sim' if difference < 0 else 'Não'}")
            print()

            f.write(f"Configuração top-{i+1}:\n")
            f.write(f"Hiperparâmetros: {top5_configs[i]}\n\n")

            f.write(f"BASELINE (Stable-Baselines3 DQN):\n")
            f.write(f"  Recompensa média: {baseline_result['mean_reward']:.4f}\n")
            f.write(f"  Desvio padrão: {baseline_result['std_reward']:.4f}\n")
            f.write(f"  Recompensas por run: {baseline_result['run_rewards']}\n\n")

            f.write(f"ORIGINAL (Custom DQN):\n")
            f.write(f"  Recompensa média: {original_result['mean_reward']:.4f}\n")
            f.write(f"  Desvio padrão: {original_result['std_reward']:.4f}\n")
            f.write(f"  Recompensas por run: {original_result['run_rewards']}\n\n")

            f.write(f"COMPARAÇÃO:\n")
            f.write(f"  Diferença (Original - Baseline): {difference:.4f}\n")
            f.write(f"  Baseline superior: {'Sim' if difference < 0 else 'Não'}\n")
            f.write(
                f"  Significância da diferença: {abs(difference) / max(baseline_result['std_reward'], original_result['std_reward']):.2f}\n"
            )
            f.write(f"{'='*80}\n\n")

    # Save baseline results as JSON for further analysis
    baseline_json = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "num_runs": num_runs,
        "results": baseline_results,
        "configurations": top5_configs,
    }

    with open("baseline_results.json", "w") as f:
        json.dump(baseline_json, f, indent=2)

    print("Resultados salvos em:")
    print("- baseline_comparison_detailed.txt (comparação detalhada)")
    print("- baseline_results.json (resultados em formato JSON)")

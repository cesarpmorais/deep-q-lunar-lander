import gymnasium as gym
import numpy as np
import os
import json
from typing import Dict, List
import itertools
from dqn import DQNAgent, train_dqn, evaluate_agent
import matplotlib.pyplot as plt


def grid_search(
    env: gym.Env,
    param_grid: Dict,
    num_episodes: int = 500,
    eval_episodes: int = 20,
    num_runs: int = 3,
    results_file_name: str = "hyperparameter_search_results.json",
):
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total de configurações a testar: {len(combinations)}")
    print(f"Cada configuração será executada {num_runs} vezes\n")

    results = []

    for idx, config in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"Testando configuração {idx + 1}/{len(combinations)}")
        print(f"{'='*60}")
        print("Hiperparâmetros:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        run_rewards = []
        run_history_files = []

        for run in range(num_runs):
            print(f"\nExecutando run {run + 1}/{num_runs}...")

            agent = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                **config,
            )

            episode_rewards, eval_rewards = train_dqn(
                agent=agent,
                env=env,
                num_episodes=num_episodes,
                eval_freq=50,
                eval_episodes=10,
                save_model=False,
                save_path=f"temp_model_{idx}_{run}.pth",
            )

            final_reward = evaluate_agent(agent, env, num_episodes=eval_episodes)
            run_rewards.append(final_reward)
            print(f"Recompensa final (run {run + 1}): {final_reward:.2f}")

            histories_dir = os.path.join("results", "hyper_search_histories")
            os.makedirs(histories_dir, exist_ok=True)
            history_fname = os.path.join(histories_dir, f"config_{idx}_run_{run}.npz")
            np.savez_compressed(
                history_fname,
                episode_rewards=np.array(episode_rewards, dtype=float),
                eval_rewards=np.array(eval_rewards, dtype=float),
                loss_history=np.array(agent.loss_history, dtype=float),
            )
            run_history_files.append(history_fname)

        mean_reward = np.mean(run_rewards)
        std_reward = np.std(run_rewards)

        result = {
            "config": config,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "run_rewards": run_rewards,
            "run_history_files": run_history_files,
        }
        results.append(result)

        print(f"\nResultado médio: {mean_reward:.2f} ± {std_reward:.2f}")

        save_search_results(results, results_file_name)

    results.sort(key=lambda x: x["mean_reward"], reverse=True)

    save_search_results(results, results_file_name)

    return results


def save_search_results(results: List[Dict], filename: str):
    results_serializable = []
    for r in results:
        result_copy = r.copy()
        result_copy["run_rewards"] = [float(x) for x in result_copy["run_rewards"]]
        result_copy["mean_reward"] = float(result_copy["mean_reward"])
        result_copy["std_reward"] = float(result_copy["std_reward"])

        if "run_history_files" in result_copy:
            result_copy["run_history_files"] = [
                str(x) for x in result_copy["run_history_files"]
            ]

        results_serializable.append(result_copy)

    with open(filename, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResultados salvos em: {filename}")


def plot_search_results(results: List[Dict], top_n: int = 10):
    """Plota os resultados da busca de hiperparâmetros"""

    top_results = results[:top_n]

    configs_str = [f"Config {i+1}" for i in range(len(top_results))]
    mean_rewards = [r["mean_reward"] for r in top_results]
    std_rewards = [r["std_reward"] for r in top_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(configs_str))

    ax.bar(x_pos, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax.set_xlabel("Configuração")
    ax.set_ylabel("Recompensa Média")
    ax.set_title(f"Top {top_n} Configurações de Hiperparâmetros")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs_str, rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("hyperparameter_search_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print(f"TOP {min(5, len(results))} MELHORES CONFIGURAÇÕES")
    print("=" * 60)

    for idx, result in enumerate(results[:5]):
        print(
            f"\n{idx + 1}. Recompensa: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}"
        )
        print("   Hiperparâmetros:")
        for key, value in result["config"].items():
            print(f"     {key}: {value}")


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")

    param_grid = {
        "learning_rate": [5e-4, 8e-4, 1e-3],
        "gamma": [0.99, 0.995],
        "epsilon_decay": [0.999, 0.9995],
        "batch_size": [32, 64],
        "target_update_freq": [5, 10],
        "hidden_dims": [[64, 64], [128, 128]],
        "epsilon_start": [1.0],
        "epsilon_end": [0.01],
        "buffer_capacity": [50000, 100000],
    }

    print("Iniciando grid search de hiperparâmetros...\n")
    results_grid = grid_search(
        env=env,
        param_grid=param_grid,
        num_episodes=500,
        eval_episodes=20,
        num_runs=3,
    )

    env.close()

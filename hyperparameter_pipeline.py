import time
import json
import os

import gymnasium as gym

from hyperparameter_search import grid_search

if __name__ == "__main__":
    # Parameters for the pipeline
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    phaseA_file = f"results/hyper_search_phaseA_{timestamp}.json"
    phaseB_file = f"results/hyper_search_phaseB_{timestamp}.json"
    phaseA_hist_dir = os.path.join("results", "hyper_search_histories_phaseA")
    phaseB_hist_dir = os.path.join("results", "hyper_search_histories_phaseB")

    env = gym.make("LunarLander-v3")

    ## 1a leva de testes com parametros
    # param_grid = {
    #     "learning_rate": [1e-4, 5e-4, 1e-3],
    #     "gamma": [0.95, 0.99],
    #     "epsilon_decay": [0.995, 0.999],
    #     "batch_size": [32, 64],
    #     "target_update_freq": [5, 10],
    #     "hidden_dims": [[64, 64], [128, 128]],
    #     "epsilon_start": [1.0],
    #     "epsilon_end": [0.01],
    #     "buffer_capacity": [50000],
    # }

    ## 2a leva de testes
    # param_grid = {
    #     "learning_rate": [5e-4, 8e-4, 1e-3],
    #     "gamma": [0.99, 0.995],
    #     "epsilon_decay": [0.999, 0.9995],
    #     "batch_size": [32, 64],
    #     "target_update_freq": [5, 10],
    #     "hidden_dims": [[64, 64], [128, 128]],
    #     "epsilon_start": [1.0],
    #     "epsilon_end": [0.01],
    #     "buffer_capacity": [50000, 100000],
    # }

    ## 3a leva de testes
    param_grid = {
        "learning_rate": [5e-4, 8e-4, 1e-3],
        "gamma": [0.99, 0.995],
        "epsilon_decay": [0.999, 0.9995],
        "batch_size": [64],
        "target_update_freq": [10],
        "hidden_dims": [[64, 64], [128, 128]],
        "epsilon_start": [1.0],
        "epsilon_end": [0.01],
        "buffer_capacity": [50000, 100000],
    }

    print("Phase A: quick triage (num_episodes=100, num_runs=1)")
    results_phaseA = grid_search(
        env=env,
        param_grid=param_grid,
        num_episodes=100,
        eval_episodes=10,
        num_runs=3,
        results_file_name=phaseA_file,
    )

    with open(phaseA_file, "r") as f:
        phaseA_data = json.load(f)

    K = 5
    top_k = phaseA_data[:K]
    top_configs = [r["config"] for r in top_k]

    print(
        f"\nPhase B: refining top {len(top_configs)} configs (num_episodes=500, num_runs=10s)"
    )

    # Para cada configuração do top-K, chame `grid_search` com um `param_grid`
    # que contém apenas essa configuração (listas de tamanho 1). Isso reaproveita
    # toda a lógica de salvamento/serialização do `grid_search` e produz um
    # JSON temporário por configuração que agregamos ao final.
    tmp_files = []
    for i, cfg in enumerate(top_configs):
        print(f"\nRefining config {i+1}/{len(top_configs)}")
        single_param_grid = {k: [v] for k, v in cfg.items()}
        tmp_fname = f"results/hyper_search_phaseB_{timestamp}_cfg{i}.json"
        tmp_files.append(tmp_fname)

        grid_search(
            env=env,
            param_grid=single_param_grid,
            num_episodes=500,
            eval_episodes=10,
            num_runs=5,
            results_file_name=tmp_fname,
        )

    # Agrega todos os resultados temporários em um único JSON final
    aggregated = []
    for tf in tmp_files:
        if os.path.exists(tf):
            with open(tf, "r") as f:
                data = json.load(f)
                aggregated.extend(data)

    aggregated.sort(key=lambda x: x["mean_reward"], reverse=True)
    os.makedirs(os.path.dirname(phaseB_file), exist_ok=True)
    with open(phaseB_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated Phase B results to: {phaseB_file}")

    os.makedirs(os.path.dirname("top5_configs"), exist_ok=True)
    with open("top5_configs", "w") as f:
        json.dump(aggregated[:5], f, indent=2)

    env.close()

    print("Pipeline finished.")

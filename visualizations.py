# visualizations.py
"""
Final DQNVisualizer-based visualizations module (OPÇÃO A - classe única)

Este arquivo contém a classe `DQNVisualizer` e um `main` que monta a
pipeline descrita no relatório:
  - Top-3 Phase A
  - Top-5 Phase B
  - Melhor configuração final
  - Comparação com baseline (SB3)
  - Configuração ruim (opcional)

Requisitos:
  - Os arquivos `.npz` gerados pela busca de hiperparâmetros devem
    conter sempre: `episode_rewards`, `eval_rewards`, `loss_history`.
  - Os arquivos JSON da Phase A/B e baseline devem seguir a estrutura
    já gerada pelos seus scripts (lista de dicionários com campos
    `config`, `mean_reward`, `std_reward`, `run_history_files`).

Uso:
    python visualizations.py

Saída (por padrão): `results/plots/` com subpastas.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class DQNVisualizer:
    def __init__(self, save_dir: str = "results/plots", style: str = "seaborn-v0_8"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Visual style
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ]

    # ------------------------------ I/O helpers ------------------------------
    def load_npz(self, path: str) -> Dict[str, np.ndarray]:
        """Carrega um .npz e garante os três arrays esperados.

        Retorna dicionário com chaves: episode_rewards, eval_rewards, loss_history.
        """
        data = np.load(path)
        out = {}
        # assumimos que esses três existem (conforme você confirmou)
        out["episode_rewards"] = data["episode_rewards"].astype(float)
        out["eval_rewards"] = data["eval_rewards"].astype(float)
        out["loss_history"] = data["loss_history"].astype(float)
        return out

    # --------------------------- plotting utilities -------------------------
    @staticmethod
    def moving_average(x: np.ndarray, window: int = 50) -> np.ndarray:
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    # --------------------------- core plotting -----------------------------
    def plot_training_progress(
        self,
        episode_rewards: np.ndarray,
        eval_rewards: Optional[np.ndarray],
        loss_history: Optional[np.ndarray],
        epsilon_history: Optional[np.ndarray],
        eval_freq: int,
        config: Optional[Dict[str, Any]] = None,
        out_path: Optional[str] = None,
        title_suffix: str = "",
    ) -> None:
        """Plota um grid com: (1) recompensas por episódio + média móvel,
        (2) recompensas de avaliação, (3) loss, (4) epsilon.

        Se `out_path` for None, salva em self.save_dir/training.
        """
        # prepara saída
        out_dir = self.save_dir / "training"
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            out_path = out_dir / f"training_progress{title_suffix}.png"
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        # layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2, ax3, ax4 = axes.ravel()

        # 1) recompensa por episódio
        ep = np.array(episode_rewards)
        ax1.plot(ep, alpha=0.3, label="Recompensa por episódio", color=self.colors[1])
        window = min(50, max(1, len(ep) // 10))
        if len(ep) >= window:
            ma = self.moving_average(ep, window)
            ax1.plot(range(window - 1, len(ep)), ma, linewidth=2, label=f"Média móvel ({window})", color=self.colors[0])
        ax1.set_title("Recompensas de Treinamento")
        ax1.set_xlabel("Episódio")
        ax1.set_ylabel("Recompensa")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2) avaliação
        if eval_rewards is not None:
            eval_x = [i * eval_freq for i in range(1, len(eval_rewards) + 1)]
            ax2.plot(eval_x, eval_rewards, marker="o", label="Avaliação", color=self.colors[2])
            ax2.set_title("Recompensa Média de Avaliação")
            ax2.set_xlabel("Episódio")
            ax2.set_ylabel("Recompensa média")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Sem eval_rewards", ha="center", va="center")
            ax2.set_axis_off()

        # 3) loss
        if loss_history is not None and len(loss_history) > 0:
            loss = np.array(loss_history)
            # tentativa de suavização
            lw = min(100, max(1, len(loss) // 10))
            if len(loss) >= lw:
                smooth = self.moving_average(loss, lw)
                ax3.plot(range(lw - 1, len(loss)), smooth, label="Loss suavizada", color=self.colors[3])
            else:
                ax3.plot(loss, label="Loss", color=self.colors[3])
            ax3.set_yscale("log")
            ax3.set_title("Loss durante o treinamento")
            ax3.set_xlabel("Atualizações")
            ax3.set_ylabel("Loss (MSE)")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "Sem loss_history", ha="center", va="center")
            ax3.set_axis_off()

        # 4) epsilon
        if epsilon_history is not None and len(epsilon_history) > 0:
            eps = np.array(epsilon_history)
            ax4.plot(eps, label="Epsilon", color=self.colors[4])
            ax4.set_title("Decaimento de Epsilon")
            ax4.set_xlabel("Atualizações")
            ax4.set_ylabel("Epsilon")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Sem epsilon_history", ha="center", va="center")
            ax4.set_axis_off()

        # info de config (pequeno box)
        if config is not None:
            cfg_items = list(config.items())[:6]
            cfg_text = "\n".join([f"{k}: {v}" for k, v in cfg_items])
            fig.text(0.02, 0.02, f"Config:\n{cfg_text}", fontsize=9, bbox=dict(facecolor="lightgray", alpha=0.5))

        fig.suptitle(f"DQN Training Progress {title_suffix}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_hyperparameter_comparison(self, results: List[Dict[str, Any]], top_n: int = 15) -> None:
        """Plota ranking das top_n configurações e importância relativa dos parâmetros."""
        out_dir = self.save_dir / "hyper"
        out_dir.mkdir(parents=True, exist_ok=True)

        top_results = results[:top_n]
        labels = [f"Cfg {i+1}" for i in range(len(top_results))]
        means = [r["mean_reward"] for r in top_results]
        stds = [r["std_reward"] for r in top_results]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Recompensa média")
        ax.set_title(f"Top {top_n} configurações (média ± std)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig_path = out_dir / "hyper_topk.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # importância relativa
        importance = self._calculate_parameter_importance(results)
        if importance:
            fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.5)))
            params = list(importance.keys())
            scores = list(importance.values())
            ax.barh(params, scores, color=self.colors[2])
            ax.set_title("Importância relativa dos hiperparâmetros")
            ax.set_xlabel("Score normalizado")
            ax.grid(True, axis="x", alpha=0.3)
            fig_path2 = out_dir / "hyper_parameter_importance.png"
            fig.savefig(fig_path2, dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _calculate_parameter_importance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}
        param_importance = {}
        all_params = set()
        for r in results:
            all_params.update(r["config"].keys())

        for param in all_params:
            values = []
            rewards = []
            for r in results:
                if param in r["config"]:
                    values.append(str(r["config"][param]))
                    rewards.append(r["mean_reward"])
            if len(set(values)) <= 1:
                continue
            df = pd.DataFrame({"value": values, "reward": rewards})
            group_means = df.groupby("value")["reward"].mean()
            importance = group_means.std() / (group_means.mean() + 1e-9)
            param_importance[param] = float(importance)

        if not param_importance:
            return {}
        max_score = max(param_importance.values())
        param_importance = {k: v / max_score for k, v in param_importance.items()}
        return dict(sorted(param_importance.items(), key=lambda x: x[1], reverse=True))

    def plot_pipeline_comparison(self, phaseA_path: str, phaseB_path: str) -> None:
        out_dir = self.save_dir / "pipeline"
        out_dir.mkdir(parents=True, exist_ok=True)

        phaseA = self.load_json_safe(phaseA_path)
        phaseB = self.load_json_safe(phaseB_path)

        # histograma fase A
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        a_rewards = [r["mean_reward"] for r in phaseA]
        ax1.hist(a_rewards, bins=20, alpha=0.7)
        ax1.axvline(np.mean(a_rewards), color="red", linestyle="--", label=f"Média {np.mean(a_rewards):.1f}")
        ax1.set_title("Fase A: distribuição das configurações")
        ax1.set_xlabel("Recompensa média")
        ax1.legend()

        # barras fase B
        b_rewards = [r["mean_reward"] for r in phaseB]
        x = np.arange(1, len(b_rewards) + 1)
        ax2.bar(x, b_rewards, alpha=0.8)
        ax2.set_title("Fase B: configurações refinadas")
        ax2.set_xlabel("Ranking")
        ax2.set_ylabel("Recompensa média")

        fig.tight_layout()
        fig_path = out_dir / "pipeline_comparison.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_baseline_comparison(self, custom_results: List[Dict[str, Any]], baseline_path: str) -> None:
        out_dir = self.save_dir / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)

        baseline = self.load_json_safe(baseline_path)

        custom_means = [r["mean_reward"] for r in custom_results[:5]]
        custom_stds = [r["std_reward"] for r in custom_results[:5]]
        baseline_means = [r["mean_reward"] for r in baseline.get("results", [])[:5]]
        baseline_stds = [r["std_reward"] for r in baseline.get("results", [])[:5]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        x = np.arange(len(custom_means))
        width = 0.35
        ax1.bar(x - width/2, custom_means, width, yerr=custom_stds, capsize=5, label="Custom DQN")
        ax1.bar(x + width/2, baseline_means, width, yerr=baseline_stds, capsize=5, label="SB3 DQN")
        ax1.set_title("Custom vs Baseline (Top-5)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Top {i+1}" for i in x])
        ax1.legend()

        # diferença
        diffs = [custom_means[i] - baseline_means[i] for i in range(len(custom_means))]
        colors = ["green" if d > 0 else "red" for d in diffs]
        ax2.bar(x, diffs, color=colors)
        ax2.axhline(0, color="black")
        ax2.set_title("Diferença (Custom - Baseline)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Top {i+1}" for i in x])

        fig.tight_layout()
        fig_path = out_dir / "baseline_comparison.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------- helpers ----------------------------
    def load_json_safe(self, path: str) -> Any:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def find_npz_for_config(self, config_index: int, folder: str = "results/hyper_search_histories") -> List[str]:
        p = Path(folder)
        if not p.exists():
            return []
        files = []
        for f in p.iterdir():
            if f.is_file() and f.name.startswith(f"config_{config_index}_") and f.suffix == ".npz":
                files.append(str(f))
        return sorted(files)

    # ---------------------------- pipeline ----------------------------
    def run_pipeline(self,
                     phaseA_glob_folder: str = "results",
                     phaseB_glob_folder: str = "results",
                     phaseA_pattern: str = "hyper_search_phaseA",
                     phaseB_pattern: str = "hyper_search_phaseB",
                     baseline_path: str = "baseline_results.json",
                     include_bad_example: bool = True,
                     eval_freq_assume: int = 50,
                     ) -> None:
        """Executa a pipeline automática: encontra os JSONs mais recentes de
        Phase A/B, seleciona top-k e gera todos os gráficos relevantes."""
        # procura os arquivos phaseA e phaseB mais recentes
        phaseA_files = [p for p in Path(phaseA_glob_folder).iterdir() if p.is_file() and phaseA_pattern in p.name]
        phaseB_files = [p for p in Path(phaseB_glob_folder).iterdir() if p.is_file() and phaseB_pattern in p.name]

        if not phaseA_files:
            print("[WARN] Nenhum arquivo Phase A encontrado em", phaseA_glob_folder)
            phaseA_results = []
        else:
            phaseA_latest = sorted(phaseA_files)[-1]
            phaseA_results = self.load_json_safe(str(phaseA_latest))

        if not phaseB_files:
            print("[WARN] Nenhum arquivo Phase B encontrado em", phaseB_glob_folder)
            phaseB_results = []
        else:
            phaseB_latest = sorted(phaseB_files)[-1]
            phaseB_results = self.load_json_safe(str(phaseB_latest))

        # Seleciona top-3 Phase A
        top3_A = phaseA_results[:3]
        # seleciona top-5 Phase B
        top5_B = phaseB_results[:5]

        # -------------------- gera plots para top3 phase A --------------------
        out_folder_A = self.save_dir / "phaseA_top3"
        out_folder_A.mkdir(parents=True, exist_ok=True)
        print("Gerando plots para TOP-3 Phase A...")
        for idx, entry in enumerate(top3_A):
            # tenta localizar os arquivos .npz gerados para essa configuração
            npz_files = self.find_npz_for_config(idx)
            for npz in npz_files:
                hist = self.load_npz(npz)
                title_suffix = f"_phaseA_cfg{idx}_run_{Path(npz).stem}"
                out_path = out_folder_A / f"{Path(npz).stem}.png"
                self.plot_training_progress(
                    episode_rewards=hist["episode_rewards"],
                    eval_rewards=hist.get("eval_rewards", None),
                    loss_history=hist.get("loss_history", None),
                    epsilon_history=None,
                    eval_freq=eval_freq_assume,
                    config=entry.get("config", {}),
                    out_path=str(out_path),
                    title_suffix=title_suffix,
                )

        # -------------------- gera plots para top5 phase B --------------------
        out_folder_B = self.save_dir / "phaseB_top5"
        out_folder_B.mkdir(parents=True, exist_ok=True)
        print("Gerando plots para TOP-5 Phase B...")
        for idx, entry in enumerate(top5_B):
            npz_files = self.find_npz_for_config(idx)
            for npz in npz_files:
                hist = self.load_npz(npz)
                title_suffix = f"_phaseB_cfg{idx}_run_{Path(npz).stem}"
                out_path = out_folder_B / f"{Path(npz).stem}.png"
                self.plot_training_progress(
                    episode_rewards=hist["episode_rewards"],
                    eval_rewards=hist.get("eval_rewards", None),
                    loss_history=hist.get("loss_history", None),
                    epsilon_history=None,
                    eval_freq=eval_freq_assume,
                    config=entry.get("config", {}),
                    out_path=str(out_path),
                    title_suffix=title_suffix,
                )

        # -------------------- melhor config final (top-1) --------------------
        print("Gerando plot para melhor configuração final (top-1 Phase B)...")
        if top5_B:
            best_entry = top5_B[0]
            best_npzs = self.find_npz_for_config(0)
            out_folder_best = self.save_dir / "best_config"
            out_folder_best.mkdir(parents=True, exist_ok=True)
            for npz in best_npzs:
                hist = self.load_npz(npz)
                out_path = out_folder_best / f"{Path(npz).stem}.png"
                self.plot_training_progress(
                    episode_rewards=hist["episode_rewards"],
                    eval_rewards=hist.get("eval_rewards", None),
                    loss_history=hist.get("loss_history", None),
                    epsilon_history=None,
                    eval_freq=eval_freq_assume,
                    config=best_entry.get("config", {}),
                    out_path=str(out_path),
                    title_suffix=f"_bestcfg_run_{Path(npz).stem}",
                )
        else:
            print("Nenhuma configuração top-1 encontrada em Phase B.")

        # -------------------- baseline comparison --------------------
        if Path(baseline_path).exists() and top5_B:
            print("Gerando comparação com baseline (Stable-Baselines3)...")
            self.plot_baseline_comparison(custom_results=top5_B, baseline_path=baseline_path)
        else:
            print("Baseline não encontrado ou top5 vazio — pulando comparação baseline.")

        # -------------------- opcional: configuração ruim --------------------
        if include_bad_example and phaseA_results:
            # ordena crescente (pior primeiro)
            worst = sorted(phaseA_results, key=lambda x: x.get("mean_reward", np.inf))[0]
            worst_idx = phaseA_results.index(worst)
            worst_npzs = self.find_npz_for_config(worst_idx)
            if worst_npzs:
                out_folder_bad = self.save_dir / "worst_config"
                out_folder_bad.mkdir(parents=True, exist_ok=True)
                print("Gerando plot para configuração ruim (exemplo didático)...")
                for npz in worst_npzs:
                    hist = self.load_npz(npz)
                    out_path = out_folder_bad / f"{Path(npz).stem}.png"
                    self.plot_training_progress(
                        episode_rewards=hist["episode_rewards"],
                        eval_rewards=hist.get("eval_rewards", None),
                        loss_history=hist.get("loss_history", None),
                        epsilon_history=None,
                        eval_freq=eval_freq_assume,
                        config=worst.get("config", {}),
                        out_path=str(out_path),
                        title_suffix=f"_worstcfg_run_{Path(npz).stem}",
                    )
            else:
                print("Nenhum .npz encontrado para a configuração pior (Phase A).")

        # -------------------- análises agregadas --------------------
        if phaseA_results:
            print("Gerando comparações agregadas da hyper search (Phase A/B)...")
            self.plot_hyperparameter_comparison(phaseA_results + phaseB_results, top_n=15)
            if phaseA_results and phaseB_results:
                # salva pipeline comparativo
                # tenta usar os dois arquivos mais recentes
                try:
                    phaseA_latest_path = str(sorted(Path(phaseA_glob_folder).iterdir())[-1])
                    phaseB_latest_path = str(sorted(Path(phaseB_glob_folder).iterdir())[-1])
                    self.plot_pipeline_comparison(phaseA_latest_path, phaseB_latest_path)
                except Exception:
                    pass

        print("Pipeline de visualizações concluída. Verifique a pasta:", self.save_dir)


# ------------------------------- CLI / Main --------------------------------
if __name__ == "__main__":
    viz = DQNVisualizer()
    viz.run_pipeline(
        phaseA_glob_folder="results",
        phaseB_glob_folder="results",
        phaseA_pattern="hyper_search_phaseA",
        phaseB_pattern="hyper_search_phaseB",
        baseline_path="baseline_results.json",
        include_bad_example=True,
        eval_freq_assume=50,
    )

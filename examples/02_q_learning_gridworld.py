"""
Ejemplo 2: Q-Learning tabular en un ambiente GridWorld.

Objetivo:
    Entrenar un agente para llegar desde S hasta G evitando huecos H. Este ejemplo
    muestra cómo se aprende una política a partir de recompensas, sin etiquetas
    supervisadas.

Ejecución:
    python examples/02_q_learning_gridworld.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Permite ejecutar este archivo desde la raíz del repositorio sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.gridworld import GridWorld
from src.q_learning import TrainingConfig, evaluate_policy, moving_average, train_q_learning


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    env = GridWorld(seed=42)
    config = TrainingConfig(
        episodes=5000,
        max_steps=100,
        alpha=0.10,
        gamma=0.95,
        epsilon_start=1.00,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=42,
    )

    q_table, history = train_q_learning(env, config)
    metrics = evaluate_policy(env, q_table, episodes=200, max_steps=100)

    # Guardar tabla Q para poder revisarla o incluirla en el informe.
    q_df = pd.DataFrame(q_table, columns=["arriba", "derecha", "abajo", "izquierda"])
    q_df.index.name = "estado"
    q_df.to_csv(output_dir / "q_table.csv")

    # Gráfico 1: recompensa suavizada.
    reward_ma = moving_average(history["reward"], window=100)
    plt.figure(figsize=(9, 5))
    plt.plot(reward_ma)
    plt.title("Q-Learning: recompensa media móvil por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa media móvil")
    plt.tight_layout()
    plt.savefig(output_dir / "q_learning_recompensa.png", dpi=160)
    plt.close()

    # Gráfico 2: tasa de éxito suavizada.
    success_ma = moving_average(history["success"], window=100)
    plt.figure(figsize=(9, 5))
    plt.plot(success_ma)
    plt.title("Q-Learning: tasa de éxito media móvil")
    plt.xlabel("Episodio")
    plt.ylabel("Tasa de éxito")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "q_learning_tasa_exito.png", dpi=160)
    plt.close()

    print(" Evaluación de política aprendida ")
    print(f"Recompensa media: {metrics['mean_reward']:.3f}")
    print(f"Pasos promedio: {metrics['mean_steps']:.2f}")
    print("\nPolítica aprendida:")
    print(env.render_policy(q_table))


if __name__ == "__main__":
    main()

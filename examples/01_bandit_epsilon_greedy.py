"""
Ejemplo 1: Bandit multi-brazo con estrategia epsilon-greedy.

Objetivo:
    Ilustrar el dilema exploración vs. explotación, una idea central en aprendizaje
    por refuerzo. El agente debe elegir entre varias máquinas tragamonedas, cada una
    con una recompensa promedio desconocida.

Ejecución:
    python examples/01_bandit_epsilon_greedy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    seed = 42
    rng = np.random.default_rng(seed)

    n_arms = 5
    n_steps = 2000
    epsilon = 0.10

    # Recompensa real promedio de cada brazo. En un problema real esto sería desconocido.
    true_means = np.array([0.2, 0.5, 0.1, 0.8, 0.4])

    # Estimaciones que el agente irá actualizando con la experiencia.
    estimated_values = np.zeros(n_arms)
    action_counts = np.zeros(n_arms)
    rewards = []

    for _step in range(n_steps):
        # Epsilon-greedy: explorar con probabilidad epsilon, explotar en caso contrario.
        if rng.random() < epsilon:
            action = int(rng.integers(n_arms))
        else:
            action = int(np.argmax(estimated_values))

        # La recompensa observada tiene ruido normal alrededor de la media real.
        reward = rng.normal(loc=true_means[action], scale=0.10)
        rewards.append(reward)

        # Actualización incremental del promedio estimado para la acción seleccionada.
        action_counts[action] += 1
        estimated_values[action] += (reward - estimated_values[action]) / action_counts[action]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    cumulative_average_reward = np.cumsum(rewards) / np.arange(1, n_steps + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(cumulative_average_reward)
    plt.title("Bandit multi-brazo: recompensa promedio acumulada")
    plt.xlabel("Paso")
    plt.ylabel("Recompensa promedio")
    plt.tight_layout()
    plt.savefig(output_dir / "bandit_recompensa_promedio.png", dpi=160)
    plt.close()

    print("Recompensas reales promedio:", true_means.round(3))
    print("Valores estimados por el agente:", estimated_values.round(3))
    print("Veces que seleccionó cada brazo:", action_counts.astype(int))
    print("Mejor brazo real:", int(np.argmax(true_means)))
    print("Mejor brazo aprendido:", int(np.argmax(estimated_values)))
    print("Gráfico guardado en outputs/bandit_recompensa_promedio.png")


if __name__ == "__main__":
    main()

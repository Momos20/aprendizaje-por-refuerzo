"""
Implementación de Q-Learning tabular.

Q-Learning aprende una función Q(s, a), que estima qué tan conveniente es tomar
una acción a en un estado s. La actualización principal es:

Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Donde:
    alpha: tasa de aprendizaje.
    gamma: factor de descuento.
    epsilon: probabilidad de explorar una acción aleatoria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.gridworld import GridWorld


@dataclass
class TrainingConfig:
    """Hiperparámetros del entrenamiento."""

    episodes: int = 5000
    max_steps: int = 100
    alpha: float = 0.10
    gamma: float = 0.95
    epsilon_start: float = 1.00
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42


def epsilon_greedy_action(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    """
    Selecciona una acción usando estrategia epsilon-greedy.

    Con probabilidad epsilon explora una acción aleatoria; de lo contrario, explota
    la mejor acción según la tabla Q.
    """
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    return int(np.argmax(q_table[state]))


def train_q_learning(env: GridWorld, config: TrainingConfig) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Entrena un agente con Q-Learning.

    Args:
        env: Ambiente GridWorld.
        config: Configuración de entrenamiento.

    Returns:
        q_table: Tabla Q aprendida.
        history: Diccionario con recompensas, éxitos y epsilon por episodio.
    """
    rng = np.random.default_rng(config.seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)

    epsilon = config.epsilon_start
    rewards_history: List[float] = []
    success_history: List[int] = []
    epsilon_history: List[float] = []

    for _episode in range(config.episodes):
        state = env.reset()
        total_reward = 0.0
        success = 0

        for _step in range(config.max_steps):
            action = epsilon_greedy_action(q_table, state, epsilon, rng)
            result = env.step(action)

            best_next_action_value = np.max(q_table[result.next_state])
            temporal_difference = result.reward + config.gamma * best_next_action_value - q_table[state, action]
            q_table[state, action] += config.alpha * temporal_difference

            state = result.next_state
            total_reward += result.reward

            if result.terminated:
                success = int(result.next_state == env.goal_state)
                break

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        rewards_history.append(total_reward)
        success_history.append(success)
        epsilon_history.append(epsilon)

    history = {
        "reward": rewards_history,
        "success": success_history,
        "epsilon": epsilon_history,
    }
    return q_table, history


def moving_average(values: List[float], window: int = 100) -> np.ndarray:
    """Calcula media móvil para suavizar una serie."""
    array = np.asarray(values, dtype=float)
    if len(array) < window:
        return array
    kernel = np.ones(window) / window
    return np.convolve(array, kernel, mode="valid")


def evaluate_policy(env: GridWorld, q_table: np.ndarray, episodes: int = 100, max_steps: int = 100) -> Dict[str, float]:
    """
    Evalúa la política greedy derivada de la tabla Q.

    Args:
        env: Ambiente GridWorld.
        q_table: Tabla Q entrenada.
        episodes: Número de episodios de evaluación.
        max_steps: Máximo de pasos por episodio.

    Returns:
        Diccionario con tasa de éxito, recompensa media y pasos promedio.
    """
    successes = 0
    rewards = []
    steps_used = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = int(np.argmax(q_table[state]))
            result = env.step(action)
            total_reward += result.reward
            state = result.next_state

            if result.terminated:
                successes += int(result.next_state == env.goal_state)
                steps_used.append(step + 1)
                break
        else:
            steps_used.append(max_steps)

        rewards.append(total_reward)

    return {
        "success_rate": successes / episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_used)),
    }

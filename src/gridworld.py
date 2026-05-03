"""
Ambiente GridWorld simple para aprendizaje por refuerzo tabular.

El objetivo es que un agente aprenda a moverse desde una casilla inicial hasta una meta,
evitando caer en huecos. Este ambiente se implementa desde cero para que el ejemplo sea
100 % reproducible y no dependa de librerías externas como Gymnasium.

Estados:
    Cada celda del tablero es un estado entero entre 0 y n_filas*n_columnas - 1.

Acciones:
    0 = arriba, 1 = derecha, 2 = abajo, 3 = izquierda.

Recompensas:
    +1.0 al llegar a la meta.
    -1.0 al caer en un hueco.
    -0.01 por cada paso, para incentivar caminos cortos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class StepResult:
    """Resultado de ejecutar una acción en el ambiente."""

    next_state: int
    reward: float
    terminated: bool


class GridWorld:
    """Ambiente discreto tipo tablero para Q-Learning."""

    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # arriba
        1: (0, 1),   # derecha
        2: (1, 0),   # abajo
        3: (0, -1),  # izquierda
    }

    ACTION_SYMBOLS: Dict[int, str] = {
        0: "↑",
        1: "→",
        2: "↓",
        3: "←",
    }

    def __init__(
        self,
        grid: List[str] | None = None,
        step_penalty: float = -0.01,
        seed: int = 42,
    ) -> None:
        """
        Inicializa el ambiente.

        Args:
            grid: Lista de cadenas con el mapa. Convenciones:
                S = inicio, F = casilla libre, H = hueco, G = meta.
            step_penalty: Penalización por paso.
            seed: Semilla para reproducibilidad.
        """
        self.grid = grid or [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG",
        ]
        self.step_penalty = step_penalty
        self.rng = np.random.default_rng(seed)

        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(self.ACTIONS)

        self.start_state = self._find_cell("S")
        self.goal_state = self._find_cell("G")
        self.hole_states = {self._to_state(r, c) for r, row in enumerate(self.grid) for c, value in enumerate(row) if value == "H"}
        self.state = self.start_state

    def reset(self) -> int:
        """Reinicia el episodio y retorna el estado inicial."""
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> StepResult:
        """
        Ejecuta una acción y retorna el nuevo estado, la recompensa y si terminó el episodio.

        Args:
            action: Entero entre 0 y 3.

        Returns:
            StepResult con next_state, reward y terminated.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Acción inválida: {action}. Debe estar entre 0 y 3.")

        row, col = self._to_position(self.state)
        d_row, d_col = self.ACTIONS[action]

        new_row = int(np.clip(row + d_row, 0, self.n_rows - 1))
        new_col = int(np.clip(col + d_col, 0, self.n_cols - 1))
        next_state = self._to_state(new_row, new_col)

        if next_state == self.goal_state:
            reward = 1.0
            terminated = True
        elif next_state in self.hole_states:
            reward = -1.0
            terminated = True
        else:
            reward = self.step_penalty
            terminated = False

        self.state = next_state
        return StepResult(next_state, reward, terminated)

    def render_policy(self, q_table: np.ndarray) -> str:
        """
        Convierte una tabla Q en una política legible con flechas.

        Args:
            q_table: Arreglo de tamaño [n_states, n_actions].

        Returns:
            Cadena con la política aprendida.
        """
        rows = []
        for r, row in enumerate(self.grid):
            symbols = []
            for c, cell in enumerate(row):
                state = self._to_state(r, c)
                if cell in {"S", "F"}:
                    best_action = int(np.argmax(q_table[state]))
                    symbols.append(self.ACTION_SYMBOLS[best_action])
                else:
                    symbols.append(cell)
            rows.append(" ".join(symbols))
        return "\n".join(rows)

    def _find_cell(self, target: str) -> int:
        """Busca una celda del mapa y retorna su estado entero."""
        for r, row in enumerate(self.grid):
            for c, value in enumerate(row):
                if value == target:
                    return self._to_state(r, c)
        raise ValueError(f"No se encontró la celda {target} en el mapa.")

    def _to_state(self, row: int, col: int) -> int:
        """Convierte coordenadas de fila/columna a estado entero."""
        return row * self.n_cols + col

    def _to_position(self, state: int) -> Tuple[int, int]:
        """Convierte un estado entero a coordenadas de fila/columna."""
        return divmod(state, self.n_cols)

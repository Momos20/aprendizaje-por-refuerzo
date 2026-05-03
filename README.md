# Aprendizaje por Refuerzo - Ejemplos reproducibles en Python

Repositorio de apoyo para la exposición **Aprendizaje por Refuerzo (Reinforcement Learning)** del curso de **Aprendizaje Automático**.

El objetivo de este repositorio es acompañar la presentación con ejemplos computacionales simples, reproducibles y bien documentados. A través del código se ilustran los conceptos centrales del aprendizaje por refuerzo:

- agente, entorno, estado, acción, recompensa y política;
- aprendizaje por ensayo y error;
- exploración vs. explotación;
- recompensa acumulada a largo plazo;
- entrenamiento de una política mediante Q-Learning tabular;
- evaluación del desempeño de un agente después del entrenamiento.

---

## 1. Estructura del repositorio

```text
aprendizaje-por-refuerzo/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── gridworld.py
│   └── q_learning.py
├── examples/
│   ├── 01_bandit_epsilon_greedy.py
│   └── 02_q_learning_gridworld.py
└── outputs/
    ├── bandit_recompensa_promedio.png
    ├── q_learning_recompensa.png
    ├── q_learning_tasa_exito.png
    └── q_table.csv
```

La carpeta `src/` contiene la lógica reutilizable del proyecto. La carpeta `examples/` contiene los scripts que se ejecutan para generar los resultados. La carpeta `outputs/` almacena las gráficas y archivos generados por los ejemplos.

---

## 2. ¿Qué hace cada archivo principal?

### `src/gridworld.py`

Este archivo define el **ambiente** donde aprende el agente.

En aprendizaje por refuerzo, el agente no aprende a partir de etiquetas como en aprendizaje supervisado, sino interactuando con un entorno. En este proyecto, el entorno es un tablero tipo **GridWorld**, donde el agente debe desplazarse desde una casilla inicial hasta una meta, evitando caer en huecos.

El mapa usado es:

```text
S F F F
F H F H
F F F H
H F F G
```

Donde:

| Símbolo | Significado |
|---|---|
| `S` | Estado inicial del agente |
| `F` | Casilla libre |
| `H` | Hueco o estado de penalización |
| `G` | Meta |

El agente puede ejecutar cuatro acciones:

| Acción | Movimiento |
|---|---|
| `0` | Arriba |
| `1` | Derecha |
| `2` | Abajo |
| `3` | Izquierda |

Las recompensas del ambiente son:

| Situación | Recompensa |
|---|---:|
| Llegar a la meta | `+1.0` |
| Caer en un hueco | `-1.0` |
| Dar un paso normal | `-0.01` |

En resumen, `gridworld.py` responde la pregunta:

> ¿En qué mundo aprende el agente y qué ocurre cuando toma una acción?

Este archivo implementa funciones como:

- `reset()`: reinicia el ambiente y ubica al agente en el estado inicial;
- `step(action)`: ejecuta una acción y retorna el nuevo estado, la recompensa y si el episodio terminó;
- `render_policy(q_table)`: convierte la tabla Q aprendida en una política visual con flechas.

---

### `src/q_learning.py`

Este archivo implementa el **algoritmo de aprendizaje**.

Aquí se encuentra la lógica de **Q-Learning tabular**, que permite que el agente aprenda qué acción conviene tomar en cada estado. El agente construye una tabla llamada **Q-table**, donde cada fila representa un estado y cada columna representa una acción.

La tabla aprende valores de la forma:

```text
Q(estado, acción)
```

Es decir, estima qué tan conveniente es ejecutar una acción específica estando en un estado determinado.

La regla de actualización usada por Q-Learning es:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
```

Donde:

| Símbolo | Significado |
|---|---|
| `s` | Estado actual |
| `a` | Acción tomada |
| `r` | Recompensa recibida |
| `s'` | Nuevo estado |
| `α` | Tasa de aprendizaje |
| `γ` | Factor de descuento |
| `max Q(s',a')` | Mejor valor esperado desde el nuevo estado |

En palabras simples, el agente compara lo que esperaba obtener con lo que realmente obtuvo y corrige su tabla Q poco a poco.

Este archivo incluye:

- `TrainingConfig`: define los hiperparámetros del entrenamiento;
- `epsilon_greedy_action()`: implementa la estrategia de exploración vs. explotación;
- `train_q_learning()`: entrena al agente usando Q-Learning;
- `moving_average()`: calcula medias móviles para suavizar las gráficas;
- `evaluate_policy()`: evalúa qué tan buena es la política aprendida.

En resumen, `q_learning.py` responde la pregunta:

> ¿Cómo aprende el agente a mejorar sus decisiones con la experiencia?

---

## 3. ¿Qué archivos se deben correr?

Los archivos que se ejecutan son los de la carpeta `examples/`.

No es necesario correr directamente:

```text
src/gridworld.py
src/q_learning.py
```

Esos dos archivos funcionan como módulos internos. Son importados por los ejemplos.

Los scripts que sí se deben correr son:

```text
examples/01_bandit_epsilon_greedy.py
examples/02_q_learning_gridworld.py
```

---

## 4. Instalación

Se recomienda usar un entorno virtual para aislar las dependencias del proyecto.

### Linux / Mac

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Ejemplo 1: Bandit multi-brazo

Este ejemplo explica el dilema **exploración vs. explotación**.

El agente tiene varias acciones posibles, llamadas brazos del bandit, pero al inicio no sabe cuál entrega mayor recompensa promedio. Por eso debe balancear dos comportamientos:

- **Explorar**: probar acciones nuevas para obtener información;
- **Explotar**: elegir la mejor acción conocida hasta el momento.

Este ejemplo se ejecuta con:

```bash
python examples/01_bandit_epsilon_greedy.py
```

Salida esperada:

```text
outputs/bandit_recompensa_promedio.png
```

Esta gráfica muestra cómo evoluciona la recompensa promedio a medida que el agente aprende qué acción suele ser más conveniente.

---

## 6. Ejemplo 2: Q-Learning en GridWorld

Este es el ejemplo principal del repositorio.

El agente debe aprender a moverse desde la casilla inicial `S` hasta la meta `G`, evitando caer en los huecos `H`. Al comienzo, el agente toma muchas decisiones aleatorias. Con el paso de los episodios, actualiza la tabla Q y aprende una política más efectiva.

Este ejemplo se ejecuta con:

```bash
python examples/02_q_learning_gridworld.py
```

Salida esperada en consola:

```text
=== Evaluación de política aprendida ===
Tasa de éxito: valor cercano a 100 %
Recompensa media: valor positivo cercano a 1
Pasos promedio: número bajo de pasos hacia la meta
```

Archivos generados:

```text
outputs/q_learning_recompensa.png
outputs/q_learning_tasa_exito.png
outputs/q_table.csv
```

La salida también muestra una política aprendida con flechas, por ejemplo:

```text
→ → ↓ ←
↓ H ↓ H
→ → ↓ H
H → → G
```

Cada flecha representa la acción que el agente considera más conveniente en cada estado.

---

## 7. Relación entre los conceptos de la exposición y el código

| Concepto de la exposición | Archivo relacionado | Explicación |
|---|---|---|
| Agente | `examples/02_q_learning_gridworld.py` | Es el programa que aprende a tomar decisiones. |
| Entorno | `src/gridworld.py` | Define el tablero, la meta, los huecos y las reglas. |
| Estado | `src/gridworld.py` | Cada casilla del tablero corresponde a un estado. |
| Acción | `src/gridworld.py` | Movimientos posibles: arriba, derecha, abajo e izquierda. |
| Recompensa | `src/gridworld.py` | Retroalimentación positiva o negativa después de cada acción. |
| Política | `src/q_learning.py` | Estrategia que indica qué acción tomar en cada estado. |
| Exploración vs. explotación | `examples/01_bandit_epsilon_greedy.py` y `src/q_learning.py` | Uso de la estrategia epsilon-greedy. |
| Q-Learning | `src/q_learning.py` | Algoritmo que actualiza la tabla Q. |
| Evaluación | `examples/02_q_learning_gridworld.py` | Calcula tasa de éxito, recompensa media y pasos promedio. |

---

## 8. Reproducibilidad

Los scripts incluyen semillas fijas para facilitar la reproducción de los resultados. Sin embargo, los resultados pueden cambiar si se modifican algunos hiperparámetros.

Parámetros importantes:

| Parámetro | Significado |
|---|---|
| `episodes` | Número de episodios de entrenamiento |
| `max_steps` | Número máximo de pasos por episodio |
| `alpha` | Tasa de aprendizaje |
| `gamma` | Factor de descuento |
| `epsilon_start` | Nivel inicial de exploración |
| `epsilon_min` | Nivel mínimo de exploración |
| `epsilon_decay` | Velocidad con la que disminuye la exploración |

Por ejemplo, aumentar `episodes` suele mejorar el aprendizaje, pero también aumenta el tiempo de ejecución. Reducir demasiado `epsilon` puede hacer que el agente deje de explorar antes de encontrar una buena política.

---

## 9. Comandos rápidos

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python examples/01_bandit_epsilon_greedy.py
python examples/02_q_learning_gridworld.py
```

Después de ejecutar ambos scripts, revisar la carpeta:

```text
outputs/
```

Allí estarán las gráficas y la tabla Q generada.

---

## 10. Cómo subirlo a GitHub

1. Crear un repositorio público en GitHub.
2. Subir todos los archivos de esta carpeta.
3. Verificar que `README.md`, `requirements.txt`, `src/`, `examples/` y `outputs/` estén incluidos.
4. Ejecutar los ejemplos desde la raíz del repositorio para confirmar que todo funciona.
5. Incluir las imágenes de `outputs/` como evidencia de ejecución.
6. Compartir el enlace del repositorio en la presentación o en la entrega del curso.

---

## 11. Interpretación para la exposición

Una forma sencilla de explicar el repositorio durante la presentación es:

> En este repositorio separamos el código en dos partes. La carpeta `src` contiene la lógica reutilizable: `gridworld.py` define el ambiente donde el agente aprende, y `q_learning.py` implementa el algoritmo de Q-Learning. La carpeta `examples` contiene los archivos que se ejecutan. Primero se corre el ejemplo del bandit para explicar exploración vs. explotación, y luego el ejemplo de GridWorld para mostrar cómo un agente aprende una política mediante recompensas y penalizaciones.

---

## 12. Créditos

Material elaborado como apoyo para la exposición de **Aprendizaje por Refuerzo** en el curso de **Aprendizaje Automático - Semestre 2026-1**.

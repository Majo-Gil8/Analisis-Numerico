# Taller 2 — Análisis Numérico (CM0844)

**Nombre:** Maria Jose Gil Herrera

Solución al Taller 2, dividido en dos puntos (programación dinámica frente a
fuerza bruta en caminos mínimos por etapas, y el regulador lineal-cuadrático con
la ecuación de Riccati). Cada punto está resuelto en un solo archivo de Python
autocontenido — no dependen entre sí, se pueden correr por separado.

## Requisitos

- Python 3.9+
- Librerías: `numpy`, `scipy`, `matplotlib` (ver `requirements.txt`)

## Cómo correr

Instalar dependencias (idealmente en un entorno virtual):

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Correr cada punto:

```bash
python Punto1.py
python Punto2.py
```

Cada script imprime su progreso en la terminal y va abriendo las gráficas en
ventanas aparte (con `plt.show()`) a medida que se generan. `Punto1.py` tarda
alrededor de un minuto porque los experimentos miden tiempos reales; `Punto2.py`
corre en pocos segundos.

Los dos archivos están divididos en celdas con el separador `# %%`, así que
también se pueden correr por partes desde Spyder o VS Code (Ctrl+Enter sobre la
celda). La primera celda de cada archivo define lo básico y hay que correrla
antes que las demás.

---

## Estructura del repositorio

```
.gitignore
Taller 2/
├── Punto1.py           Fuerza bruta + programación dinámica + experimentos + Dijkstra
├── Punto2.py           Riccati con integrador propio + RK45 + trayectoria óptima
├── requirements.txt
└── README.md
```
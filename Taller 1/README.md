# Taller 1 — Análisis Numérico (CM0844)

**Nombre:** Maria Jose Gil Herrera

Solución al Taller 1, dividido en tres puntos (uno por cada tema del capítulo 1
del curso: programación lineal, optimización no lineal sin restricciones, y
optimización con restricciones). Cada punto está resuelto en un solo archivo
de Python autocontenido — no dependen entre sí, se pueden correr por separado.

## Requisitos

- Python 3.9+
- Librerías: `numpy`, `matplotlib` (ver `requirements.txt`)

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
python Punto3.py
```

Cada script imprime su progreso en la terminal y va abriendo las gráficas en
ventanas aparte (con `plt.show()`) a medida que se generan.

---

## Estructura del repositorio

```
.gitignore
Taller 1/
├── Punto1.py          Simplex + florista + extensión + sensibilidad Monte Carlo
├── Punto2.py           Gradiente / Newton amortiguado / BFGS sobre Rosenbrock
├── Punto3.py           Penalización exterior + BFGS
├── requirements.txt
└── README.md
```

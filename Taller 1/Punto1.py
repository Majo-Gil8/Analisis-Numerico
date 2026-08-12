"""
Punto 1 - Programacion lineal: metodo Simplex
Analisis Numerico (CM0844) - Taller 1
Nombre: Maria Jose Gil Herrera

"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


def simplex_max(c, A, b, verbose=True):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    m, n = A.shape

    if np.any(b < 0):
        raise ValueError("Se requiere b >= 0 (arranca de la base de holguras).")

    n_total = n + m
    T = np.zeros((m + 1, n_total + 1))
    T[:m, :n] = A
    T[:m, n:n_total] = np.eye(m)
    T[:m, -1] = b
    T[m, :n] = -c

    basis = list(range(n, n_total))
    n_iter = 0

    def extraer_x():
        x = np.zeros(n)
        for row, var in enumerate(basis):
            if var < n:
                x[var] = T[row, -1]
        return x

    def nombre(idx):
        return f"x{idx+1}" if idx < n else f"s{idx-n+1}"

    path = [(extraer_x().copy(), T[m, -1])]
    if verbose:
        print(f"  Iter 0 (inicio): x = {np.round(path[0][0], 3)}   Z = {path[0][1]:.3f}   "
              f"(base = holguras: {[nombre(v) for v in basis]})")

    while True:
        obj_row = T[m, :n_total]
        if np.all(obj_row >= -1e-9):
            break  # ya no hay costo reducido negativo, se acabo

        entering = int(np.argmin(obj_row))
        col = T[:m, entering]
        rhs = T[:m, -1]
        ratios = np.where(col > 1e-9, rhs / np.where(col > 1e-9, col, 1), np.inf)
        if np.all(np.isinf(ratios)):
            raise RuntimeError("Problema no acotado.")
        leaving_row = int(np.argmin(ratios))
        leaving_var = basis[leaving_row]

        pivot = T[leaving_row, entering]
        T[leaving_row, :] /= pivot
        for i in range(m + 1):
            if i != leaving_row:
                T[i, :] -= T[i, entering] * T[leaving_row, :]
        basis[leaving_row] = entering
        n_iter += 1

        x_actual = extraer_x().copy()
        z_actual = T[m, -1]
        path.append((x_actual, z_actual))

        if verbose:
            print(f"  Iter {n_iter}: entra {nombre(entering)}, sale {nombre(leaving_var)}  ->  "
                  f"x = {np.round(x_actual, 3)}   Z = {z_actual:.3f}")

        if n_iter > 200:
            raise RuntimeError("Demasiadas iteraciones (posible ciclo).")

    x = path[-1][0]
    z = path[-1][1]

    return {"x": x, "z": z, "tableau": T, "basis": basis, "n_iter": n_iter, "path": path}


# autoprueba rapida, problema trivial donde ya se sabe la respuesta
_r = simplex_max(c=[1, 1], A=[[1, 0], [0, 1]], b=[2, 3], verbose=False)
assert np.allclose(_r["x"], [2, 3]) and np.isclose(_r["z"], 5)
print("Autoprueba de simplex_max: OK (x*=[2,3], Z*=5)\n")


def vertices_poligono(A, b):
    # vertices del poligono factible en 2D
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    A_ext = np.vstack([A, [-1, 0], [0, -1]])
    b_ext = np.concatenate([b, [0, 0]])
    n_restr = len(b_ext)
    puntos = []
    for i in range(n_restr):
        for j in range(i + 1, n_restr):
            M = A_ext[[i, j]]
            if abs(np.linalg.det(M)) < 1e-9:
                continue  # rectas paralelas, no se cruzan
            p = np.linalg.solve(M, b_ext[[i, j]])
            if np.all(A_ext @ p <= b_ext + 1e-7):
                puntos.append((round(float(p[0]), 4), round(float(p[1]), 4)))
    return sorted(set(puntos))


# Problema de la florista
print("problema de la florista")

c_flor = [2000, 1000]
A_flor = [[3, 1],
          [1, 1],
          [1, 3]]
b_flor = [300, 140, 300]

print("Trayectoria del Simplex (por iteracion):")
r2 = simplex_max(c_flor, A_flor, b_flor)
print()
print("x* =", r2["x"])
print("Z* =", r2["z"])
print("Iteraciones:", r2["n_iter"])


# grafica: region factible + trayectoria del simplex
fig, ax = plt.subplots(figsize=(6, 5))
x1_plot = np.linspace(-5, 160, 400)
rectas = {
    "3x1+x2=300 (rosas)": (300 - 3 * x1_plot),
    "x1+x2=140 (tulipanes)": (140 - x1_plot),
    "x1+3x2=300 (hibiscos)": (300 - x1_plot) / 3,
}
for nombre_r, y in rectas.items():
    ax.plot(x1_plot, y, label=nombre_r, lw=1.5)

verts = vertices_poligono(A_flor, b_flor)
vx = [v[0] for v in verts] + [verts[0][0]]
vy = [v[1] for v in verts] + [verts[0][1]]
ax.fill(vx, vy, color="lightblue", alpha=0.5, zorder=0, label="Region factible")

path_x = [p[0][0] for p in r2["path"]]
path_y = [p[0][1] for p in r2["path"]]
ax.plot(path_x, path_y, "o-", color="red", lw=2, ms=8, label="Trayectoria del Simplex")
for i, (px, py) in enumerate(zip(path_x, path_y)):
    ax.annotate(f"  iter {i}\n  ({px:.0f},{py:.0f})", (px, py), fontsize=8)

ax.set_xlim(-5, 160)
ax.set_ylim(-5, 160)
ax.set_xlabel("x1 (ramos de rosas)")
ax.set_ylabel("x2 (ramos de tulipanes)")
ax.set_title("Region factible y trayectoria del Simplex")
ax.legend(fontsize=7, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
print("Mostrando: region factible y trayectoria del Simplex\n")


# Agregando un tercer arreglo floral
print("Agregando un tercer arreglo floral")

# x3 = "ramo nuevo": 1 rosa, 1 tulipan, 4 hibiscos por unidad, precio 1200
c_ext = [2000, 1000, 1200]
A_ext = [[3, 1, 1],    # rosas
         [1, 1, 1],    # tulipanes
         [1, 3, 4]]    # hibiscos
b_ext = [300, 140, 300]

print("Nuevo arreglo x3 = 'ramo nuevo': 1 rosa, 1 tulipan, 4 hibiscos; precio 1200\n")

print(f"  max Z = {c_ext[0]}x1 + {c_ext[1]}x2 + {c_ext[2]}x3")
print("  s.a.")
print(f"      {A_ext[0][0]}x1 + {A_ext[0][1]}x2 + {A_ext[0][2]}x3 <= {b_ext[0]}   (rosas)")
print(f"      {A_ext[1][0]}x1 + {A_ext[1][1]}x2 + {A_ext[1][2]}x3 <= {b_ext[1]}   (tulipanes)")
print(f"      {A_ext[2][0]}x1 + {A_ext[2][1]}x2 + {A_ext[2][2]}x3 <= {b_ext[2]}   (hibiscos)")
print("      x1, x2, x3 >= 0\n")

print("Trayectoria del Simplex (por iteracion):")
r3 = simplex_max(c_ext, A_ext, b_ext)
print()
print("x* =", r3["x"])
print("Z* =", r3["z"])
print("Iteraciones:", r3["n_iter"])
print(f"""x3* = {r3['x'][2]:.2f} > 0, o sea que SI conviene producir este ramo. Z* subio de
220000 a {r3['z']:.0f}: agregar este ramo mejora el resultado de la florista.""")


# Sensibilidad por Monte Carlo
# se perturban b y c juntos con ruido normal alrededor de su valor nominal.
# sigma de cada b_i y c_j = un porcentaje de su valor nominal. Los b negativos que puedan salir del
# muestreo se recortan a un minimo positivo porque simplex_max exige b>=0.

print("Sensibilidad por Monte Carlo")

np.random.seed(42)  # para que el resultado sea reproducible

PORC_SIGMA = 0.30   # 30% de cada b_i y c_j nominal (modifica el ancho de la campana normal de muestreo)
N_SIM = 2000

b0 = np.array(b_flor, dtype=float)
c0 = np.array(c_flor, dtype=float)
sigma_b = PORC_SIGMA * b0
sigma_c = PORC_SIGMA * c0

print(f"b0 = {b0}   (sigma_b = {sigma_b}, {PORC_SIGMA:.0%} de cada uno)")
print(f"c0 = {c0}   (sigma_c = {sigma_c}, {PORC_SIGMA:.0%} de cada uno)")
print(f"N = {N_SIM} simulaciones\n")

basis_nominal = tuple(sorted(r2["basis"]))
verts_nominal = vertices_poligono(A_flor, b_flor)

# antes de simular calculamos con las formulas del tableau el rango teorico de cada b_i que mantiene la base actual
print("Rangos teoricos de cada b_i que mantienen la base optima:")
m, n = len(b_flor), len(c_flor)
T_teo = r2["tableau"]
xB_teo = T_teo[:m, -1]
rangos_b_teoricos = []
for i in range(m):
    g = T_teo[:m, n + i]
    cand_max = [-xB_teo[j] / g[j] for j in range(m) if g[j] > 1e-9]
    cand_min = [-xB_teo[j] / g[j] for j in range(m) if g[j] < -1e-9]
    dmin = max(cand_max) if cand_max else -np.inf
    dmax = min(cand_min) if cand_min else np.inf
    lo = b_flor[i] + dmin if np.isfinite(dmin) else -np.inf
    hi = b_flor[i] + dmax if np.isfinite(dmax) else np.inf
    rangos_b_teoricos.append((lo, hi))
nombres_b = ["b1 (rosas)", "b2 (tulipanes)", "b3 (hibiscos)"]
for i, (lo, hi) in enumerate(rangos_b_teoricos):
    uso = sum(A_flor[i][k] * r2["x"][k] for k in range(n))
    holgura = b_flor[i] - uso
    estado = "ACTIVA (holgura=0)" if abs(holgura) < 1e-6 else f"NO activa (sobran {holgura:.0f})"
    print(f"  {nombres_b[i]}: rango valido [{lo:.0f}, {hi:.0f}]   -> {estado}")
print()

sim_b, sim_c, sim_x, sim_z, sim_basis_igual = [], [], [], [], []

for _ in range(N_SIM):
    b_s = np.random.normal(b0, sigma_b)
    b_s = np.clip(b_s, 1.0, None)  # que no salga negativo, simplex_max lo necesita
    c_s = np.random.normal(c0, sigma_c)
    r = simplex_max(c_s, A_flor, b_s, verbose=False)
    sim_b.append(b_s)
    sim_c.append(c_s)
    sim_x.append(r["x"])
    sim_z.append(r["z"])
    sim_basis_igual.append(tuple(sorted(r["basis"])) == basis_nominal)

sim_b = np.array(sim_b)
sim_c = np.array(sim_c)
sim_x = np.array(sim_x)
sim_z = np.array(sim_z)
sim_basis_igual = np.array(sim_basis_igual)

# que tan seguido cambia la base
pct_igual = sim_basis_igual.mean() * 100
print("(a) estabilidad de la base optima")
print(f"  {pct_igual:.1f}% de las simulaciones mantuvieron la misma base optima "
      f"que el problema nominal (vertice (80,60), rosas+tulipanes activas)")
print(f"  {100-pct_igual:.1f}% terminaron en una base distinta (otro vertice)")

# distribucion de Z*
print("\n (b) distribucion de Z*")
print(f"  Z* nominal                = {r2['z']:.1f}")
print(f"  Z* promedio (Monte Carlo) = {sim_z.mean():.1f}")
print(f"  Z* desviacion estandar    = {sim_z.std():.1f}")
print(f"  Z* minimo / maximo        = {sim_z.min():.1f} / {sim_z.max():.1f}")
print(f"  Percentil 5% / 95%        = {np.percentile(sim_z,5):.1f} / {np.percentile(sim_z,95):.1f}")

# vertices y poligono comparados con el nominal
print("\n(c) vertices y poligono, comparados con el nominal")
print(f"  Vertices del poligono nominal (b sin perturbar): {verts_nominal}")
print(f"  Vertice optimo nominal: x* = (80, 60)\n")


def mostrar_simulacion(i, etiqueta):
    verts_i = vertices_poligono(A_flor, sim_b[i])
    dist = np.linalg.norm(sim_x[i] - r2["x"])
    print(f"  [{etiqueta}]  Simulacion #{i}")
    print(f"    b = {np.round(sim_b[i],1)}   c = {np.round(sim_c[i],1)}")
    print(f"    vertices del poligono perturbado:")
    for v in verts_i:
        print(f"        {v}")
    print(f"    x* de esta simulacion = {np.round(sim_x[i],2)}   "
          f"(distancia al optimo nominal (80,60) = {dist:.2f})")
    print(f"    base: {'Igual a la nominal (rosas+tulipanes activas)' if sim_basis_igual[i] else 'Distinta a la nominal (otra esquina)'}\n")


# buscamos entre las N_SIM, cual dio el resultado mas parecido al nominal y cual el mas distinto, para verlos concretamente
distancias = np.linalg.norm(sim_x - r2["x"], axis=1)
idx_mas_parecida = int(np.argmin(distancias))
idx_mas_distinta = int(np.argmax(distancias))

mostrar_simulacion(idx_mas_parecida, "LA MAS PARECIDA al nominal")
mostrar_simulacion(idx_mas_distinta, "LA MAS DISTINTA al nominal")
mostrar_simulacion(0, "ejemplo #0")
mostrar_simulacion(1, "ejemplo #1")


# graficas de Monte Carlo 

# histograma de Z*
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.hist(sim_z, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(r2["z"], color="red", ls="--", lw=2, label=f"Z* nominal = {r2['z']:.0f}")
ax.set_xlabel("Z* (ingreso optimo de cada simulacion)")
ax.set_ylabel("Frecuencia")
ax.set_title(f"Distribucion de Z* en {N_SIM} simulaciones Monte Carlo")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# vertices optimos de todas las simulaciones sobre el poligono nominal
fig, ax = plt.subplots(figsize=(7, 5.5))
verts_n = verts_nominal
vx = [v[0] for v in verts_n] + [verts_n[0][0]]
vy = [v[1] for v in verts_n] + [verts_n[0][1]]
ax.fill(vx, vy, color="lightblue", alpha=0.35, zorder=0, label="Poligono nominal")
ax.plot(vx, vy, color="steelblue", lw=1.5)

mask_igual = sim_basis_igual
ax.scatter(sim_x[mask_igual, 0], sim_x[mask_igual, 1], s=10, color="green", alpha=0.4,
           label=f"misma base que nominal ({mask_igual.sum()})")
ax.scatter(sim_x[~mask_igual, 0], sim_x[~mask_igual, 1], s=10, color="red", alpha=0.6,
           label=f"base distinta ({(~mask_igual).sum()})")
ax.plot(80, 60, "*", color="black", ms=18, label="Optimo nominal (80,60)")

ax.set_xlabel("x1 (ramos de rosas)")
ax.set_ylabel("x2 (ramos de tulipanes)")
ax.set_title("Vertices optimos de cada simulacion vs poligono nominal")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# algnos poligonos perturbados superpuestos al nominal
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.fill(vx, vy, color="lightblue", alpha=0.4, zorder=1, label="Poligono nominal")
ax.plot(vx, vy, color="steelblue", lw=2.5, zorder=2)

np.random.seed(7)
idx_muestra = np.random.choice(N_SIM, size=15, replace=False)
for i in idx_muestra:
    verts_i = vertices_poligono(A_flor, sim_b[i])
    vxi = [v[0] for v in verts_i] + [verts_i[0][0]]
    vyi = [v[1] for v in verts_i] + [verts_i[0][1]]
    ax.plot(vxi, vyi, color="gray", lw=0.8, alpha=0.6, zorder=0)

ax.plot([], [], color="gray", lw=0.8, label="15 poligonos perturbados (muestra)")
ax.set_xlabel("x1 (ramos de rosas)")
ax.set_ylabel("x2 (ramos de tulipanes)")
ax.set_title("El poligono cambia de forma con cada perturbacion de b?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
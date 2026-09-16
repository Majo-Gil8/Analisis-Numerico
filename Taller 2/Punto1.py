"""
Punto 1 - Programacion dinamica vs fuerza bruta: caminos minimos por etapas
Analisis Numerico (CM0844) - Taller 2
Nombre: Maria Jose Gil Herrera
"""
# %% Grafo por etapas
# el grafo se guarda denso, una ruta es la tupla (j1,...,jN)
#   entrada[j] = S -> (1,j),  internos[i,a,b] = (i+1,a) -> (i+2,b),
#   salida[a] = (N,a) -> T
import itertools
import math
import time
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
SEMILLA = 2026

def generar_grafo(N, k, semilla=None, costo_max=10.0):
    rng = np.random.default_rng(semilla)   # semilla fija para reproducibilidad
    return {"N": N, "k": k,
            "entrada": rng.uniform(0, costo_max, size=k),
            "internos": rng.uniform(0, costo_max, size=(N - 1, k, k)),
            "salida": rng.uniform(0, costo_max, size=k)}

def n_aristas(g):
    return 2 * g["k"] + (g["N"] - 1) * g["k"] ** 2

_g = generar_grafo(4, 3, semilla=SEMILLA)
print(f"Grafo N=4, k=3: {3 ** 4} rutas, {n_aristas(_g)} aristas\n")
# %% Fuerza bruta
# se evaluan las k^N rutas, a N+1 sumas cada una: Theta(N k^N)

def ruta_minima_fuerza_bruta(grafo):
    N, k = grafo["N"], grafo["k"]
    entrada, internos, salida = grafo["entrada"], grafo["internos"], grafo["salida"]

    mejor_costo, mejor_ruta, operaciones = math.inf, (), 0
    for ruta in itertools.product(range(k), repeat=N):
        costo = entrada[ruta[0]]
        for i in range(N - 1):
            costo += internos[i, ruta[i], ruta[i + 1]]
        costo += salida[ruta[-1]]
        operaciones += N + 1               # N sumas + 1 comparacion
        if costo < mejor_costo:
            mejor_costo, mejor_ruta = costo, ruta

    return {"costo": float(mejor_costo), "ruta": mejor_ruta, "operaciones": operaciones}

_r = ruta_minima_fuerza_bruta(generar_grafo(5, 3, semilla=SEMILLA))
assert _r["operaciones"] == 6 * 3 ** 5
print(f"Fuerza bruta N=5, k=3: costo {_r['costo']:.6f}, ruta {_r['ruta']}, "
      f"{_r['operaciones']} operaciones\n")
# %% Programacion dinamica (induccion hacia atras)
# recursion de Bellman hacia atras sobre J_i(a), el costo optimo desde
# (etapa i, nodo a) hasta T. Cuesta (N-1)k^2+2k = |E|, una pasada por arista
#   J_N(a) = c((N,a), T)
#   J_i(a) = min_b { c((i,a),(i+1,b)) + J_{i+1}(b) },   i = N-1, ..., 1
#   J*     = min_a { c(S,(1,a)) + J_1(a) }

def ruta_minima_pd(grafo):
    N, k = grafo["N"], grafo["k"]
    entrada, internos, salida = grafo["entrada"], grafo["internos"], grafo["salida"]

    costo_por_recorrer = [float(salida[a]) for a in range(k)]   # J_N
    politica = [[0] * k for _ in range(N - 1)]
    operaciones = k

    for i in range(N - 2, -1, -1):
        bloque = internos[i]
        nuevo = [0.0] * k
        for a in range(k):
            fila = bloque[a]
            mejor, mejor_b = math.inf, 0
            for b in range(k):
                candidato = fila[b] + costo_por_recorrer[b]
                if candidato < mejor:
                    mejor, mejor_b = candidato, b
            nuevo[a] = mejor
            politica[i][a] = mejor_b
        operaciones += k * k
        costo_por_recorrer = nuevo

    mejor_costo, primer_nodo = math.inf, 0
    for a in range(k):
        candidato = float(entrada[a]) + costo_por_recorrer[a]
        if candidato < mejor_costo:
            mejor_costo, primer_nodo = candidato, a
    operaciones += k

    ruta = [primer_nodo]                    # reconstruccion con la politica
    for i in range(N - 1):
        ruta.append(politica[i][ruta[-1]])

    return {"costo": float(mejor_costo), "ruta": tuple(ruta), "operaciones": operaciones}

_g = generar_grafo(5, 3, semilla=SEMILLA)
_fb, _pd = ruta_minima_fuerza_bruta(_g), ruta_minima_pd(_g)
assert np.isclose(_fb["costo"], _pd["costo"])
print(f"PD N=5, k=3: costo {_pd['costo']:.6f}, coincide con fuerza bruta")
print(f"  operaciones: fuerza bruta {_fb['operaciones']}, PD {_pd['operaciones']} "
      f"(= |E| = {n_aristas(_g)})\n")
# %% Experimentos
# Se mide tiempo y operaciones, en dos barridos: uno en N y otro en k
PRESUPUESTO = 3.0      # segundos antes de apagar la fuerza bruta

def cronometrar(funcion, grafo, repeticiones=5, presupuesto_s=PRESUPUESTO):
    mejor, acumulado, resultado = math.inf, 0.0, None
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        resultado = funcion(grafo)
        dt = time.perf_counter() - t0
        mejor = min(mejor, dt)
        acumulado += dt
        if acumulado > presupuesto_s:
            break
    return mejor, resultado

def ajustar_orden(x, t, modelo, descartar=0):
    # "exponencial" ajusta log2(t) vs x, "potencia" ajusta log(t) vs log(x).
    # descartar quita los tamanos donde el tiempo lo domina el costo de llamar
    x, t = np.asarray(x, float), np.asarray(t, float)
    orden = np.argsort(x)
    x, t = x[orden][descartar:], t[orden][descartar:]
    ab, orde = (x, np.log2(t)) if modelo == "exponencial" else (np.log(x), np.log(t))
    pendiente, intercepto = np.polyfit(ab, orde, 1)
    res = orde - (pendiente * ab + intercepto)
    r2 = 1 - np.sum(res ** 2) / np.sum((orde - orde.mean()) ** 2)
    return float(pendiente), float(r2)

print("Verificacion de correctitud")

# caso a mano N=2 k=2: (0,0)=16, (0,1)=23, (1,0)=21, (1,1)=25 -> minimo 16
g_manual = {"N": 2, "k": 2,
            "entrada": np.array([1.0, 4.0]),
            "internos": np.array([[[5.0, 2.0], [7.0, 1.0]]]),
            "salida": np.array([10.0, 20.0])}
for f in (ruta_minima_fuerza_bruta, ruta_minima_pd):
    r = f(g_manual)
    assert np.isclose(r["costo"], 16.0) and r["ruta"] == (0, 0)
print("  caso calculado a mano (N=2, k=2): los dos dan 16.0 por la ruta (0,0)")

for N, k in [(1, 5), (2, 4), (4, 3), (6, 4), (8, 2), (3, 7)]:
    g = generar_grafo(N, k, semilla=SEMILLA + N * k)
    fb, pd_ = ruta_minima_fuerza_bruta(g), ruta_minima_pd(g)
    assert np.isclose(fb["costo"], pd_["costo"])
    assert fb["operaciones"] == (N + 1) * k ** N
    assert pd_["operaciones"] == (N - 1) * k * k + 2 * k == n_aristas(g)
print("  6 grafos aleatorios: mismo costo optimo, y el conteo reproduce")
print("  (N+1)k^N y (N-1)k^2+2k = |E|")
print("  los experimentos de abajo vuelven a comparar los dos costos en cada grafo\n")

def graficar_experimento(x, t_fb, t_pd, o_fb, o_pd, etiqueta_x, titulo, escala_x,
                         x_o=None, etiquetas=("Tiempo [s]", "Operaciones"),
                         etiqueta_x_o=None):
    x_o = x if x_o is None else x_o
    marcas = len(x) <= 40           # con muchos puntos los marcadores saturan
    fig, (ax_t, ax_o) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, x_eje, y_fb, y_pd, etiqueta_y, etiq_x in (
            (ax_t, x, t_fb, t_pd, etiquetas[0], etiqueta_x),
            (ax_o, x_o, o_fb, o_pd, etiquetas[1], etiqueta_x_o or etiqueta_x)):
        ax.plot(x_eje, y_fb, "o-" if marcas else "-", color="#c0392b", ms=5,
                label="Fuerza bruta")
        ax.plot(x_eje, y_pd, "s-" if marcas else "-", color="#2471a3", ms=5,
                label="Prog. dinamica")
        ax.set_yscale("log")
        ax.set_xscale(escala_x)
        ax.set(xlabel=etiq_x, ylabel=etiqueta_y, title=etiqueta_y)
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(fontsize=9)
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()

print("Experimento 1 - se fija k y se aumenta N\n")

K_FIJO, N_MAX = 3, 12
Ns, t_fb_1, t_pd_1, ops_fb_1, ops_pd_1 = [], [], [], [], []
for N in range(1, N_MAX + 1):
    g = generar_grafo(N, K_FIJO, semilla=SEMILLA + N)
    dt_pd, r_pd = cronometrar(ruta_minima_pd, g)
    dt_fb, r_fb = cronometrar(ruta_minima_fuerza_bruta, g)
    assert np.isclose(r_fb["costo"], r_pd["costo"])
    Ns.append(N)
    t_fb_1.append(dt_fb)
    t_pd_1.append(dt_pd)
    ops_fb_1.append(r_fb["operaciones"])
    ops_pd_1.append(r_pd["operaciones"])
    if dt_fb > PRESUPUESTO:
        break

print(f"  {'N':>3} {'t_FB [s]':>11} {'t_PD [s]':>11} {'ops FB':>12} {'ops PD':>8} "
      f"{'FB/PD':>9}")
for i, N in enumerate(Ns):
    print(f"  {N:>3} {t_fb_1[i]:>11.6f} {t_pd_1[i]:>11.6f} {ops_fb_1[i]:>12,} "
          f"{ops_pd_1[i]:>8} {ops_fb_1[i] / ops_pd_1[i]:>8.0f}x")

graficar_experimento(Ns, t_fb_1, t_pd_1, ops_fb_1, ops_pd_1,
                     f"N (numero de etapas), con k = {K_FIJO} fijo",
                     "Experimento 1: crecimiento en N", "linear")
print()

# el factor (N+1) infla la pendiente, se normaliza para aislar k^N
p_fb_1, r2_fb_1 = ajustar_orden(Ns, t_fb_1, "exponencial", descartar=4)
t_norm = np.array(t_fb_1) / (np.array(Ns) + 1)
p_norm, r2_norm = ajustar_orden(Ns, t_norm, "exponencial", descartar=4)

print("Experimento 2 - se fija N y se aumenta k\n")

N_FIJO, K_MAX = 5, 12
ks, t_fb_2, t_pd_2, ops_fb_2, ops_pd_2 = [], [], [], [], []
for k in range(2, K_MAX + 1):
    g = generar_grafo(N_FIJO, k, semilla=SEMILLA + k)
    dt_pd, r_pd = cronometrar(ruta_minima_pd, g)
    dt_fb, r_fb = cronometrar(ruta_minima_fuerza_bruta, g)
    assert np.isclose(r_fb["costo"], r_pd["costo"])
    ks.append(k)
    t_fb_2.append(dt_fb)
    t_pd_2.append(dt_pd)
    ops_fb_2.append(r_fb["operaciones"])
    ops_pd_2.append(r_pd["operaciones"])
    if dt_fb > PRESUPUESTO:
        break

print(f"  {'k':>3} {'t_FB [s]':>11} {'t_PD [s]':>11} {'ops FB':>12} {'ops PD':>8} "
      f"{'FB/PD':>9}")
for i, k in enumerate(ks):
    print(f"  {k:>3} {t_fb_2[i]:>11.6f} {t_pd_2[i]:>11.6f} {ops_fb_2[i]:>12,} "
          f"{ops_pd_2[i]:>8} {ops_fb_2[i] / ops_pd_2[i]:>8.0f}x")

graficar_experimento(ks, t_fb_2, t_pd_2, ops_fb_2, ops_pd_2,
                     f"k (nodos por etapa), con N = {N_FIJO} fijo",
                     "Experimento 2: crecimiento en k", "log")

p_fb_2, r2_fb_2 = ajustar_orden(ks, t_fb_2, "potencia", descartar=3)
p_pd_2, r2_pd_2 = ajustar_orden(ks, t_pd_2, "potencia", descartar=3)
o_fb_2, _ = ajustar_orden(ks, ops_fb_2, "potencia")
o_pd_2, _ = ajustar_orden(ks, ops_pd_2, "potencia")

# se invierte la pregunta: en vez de fijar el tamano y medir el tiempo, se fija
# un presupuesto y se mide hasta que N llega cada metodo. Asi los dos se miden
print("Experimento 3 - que tamano alcanza cada metodo con un presupuesto dado\n")

# escalera de tiempos de la PD; la de la fuerza bruta es la del experimento 1
Ns_pd = [50, 200, 1000, 5000, 20000, 80000, 300000, 800000]
t_pd_esc = []
for N in Ns_pd:
    g = generar_grafo(N, K_FIJO, semilla=SEMILLA)
    dt, r = cronometrar(ruta_minima_pd, g, repeticiones=2, presupuesto_s=8.0)
    t_pd_esc.append(dt)

def n_por_tiempo(presupuesto, Ns_med, t_med):
    # se interpola dentro del rango medido, no se extrapola
    return float(np.interp(np.log(presupuesto), np.log(t_med), Ns_med))

def n_por_operaciones(presupuesto, exacto_fb):
    # aqui no hay medicion: las dos formulas de conteo son exactas
    if exacto_fb:
        N = 1
        while (N + 2) * K_FIJO ** (N + 1) <= presupuesto:
            N += 1
        return N
    return (presupuesto - 2 * K_FIJO) / K_FIJO ** 2 + 1

PRESUPUESTOS = [1e-3, 1e-2, 1e-1, 1.0]
OPERACIONES = [1e3, 1e4, 1e5, 1e6, 1e7]
n_fb_t = [n_por_tiempo(b, Ns, t_fb_1) for b in PRESUPUESTOS]
n_pd_t = [n_por_tiempo(b, Ns_pd, t_pd_esc) for b in PRESUPUESTOS]
n_fb_o = [n_por_operaciones(b, True) for b in OPERACIONES]
n_pd_o = [n_por_operaciones(b, False) for b in OPERACIONES]

print(f"  {'presupuesto':>12} {'N fuerza bruta':>16} {'N prog. dinamica':>18} {'razon':>10}")
for b, nf, np_ in zip(PRESUPUESTOS, n_fb_t, n_pd_t):
    print(f"  {b:>10.3f} s {nf:>16.1f} {np_:>18,.0f} {np_ / nf:>9,.0f}x")

graficar_experimento(PRESUPUESTOS, n_fb_t, n_pd_t, n_fb_o, n_pd_o,
                     "presupuesto de tiempo [s]",
                     "Experimento 3: tamano alcanzable con un presupuesto dado", "log",
                     x_o=OPERACIONES,
                     etiquetas=("N alcanzado (tiempo medido)",
                                "N alcanzado (operaciones, exacto)"),
                     etiqueta_x_o="presupuesto de operaciones")
print()

# discusion del punto 1.3
factor = t_fb_1[-1] / t_pd_1[-1]
p_pd_3, _ = ajustar_orden(Ns_pd, t_pd_esc, "potencia")

print(f"""Discusion del punto 1.3

La fuerza bruta es Theta(N k^N) y hace exactamente (N+1)k^N operaciones. La PD es
Theta(N k^2) y hace (N-1)k^2 + 2k, que es el numero de aristas del grafo, o sea
que revisa cada arista una sola vez.

  exp.  cantidad ajustada         medido   teoria
   1    log2(t/(N+1)) vs N        {p_norm:6.3f}  {np.log2(K_FIJO):6.3f}
   2    log(ops) vs log(k), FB    {o_fb_2:6.3f}  {float(N_FIJO):6.3f}
   2    log(ops) vs log(k), PD    {o_pd_2:6.3f}  {2.0:6.3f}
   3    log(t) vs log(N), PD      {p_pd_3:6.3f}  {1.0:6.3f}

Los ajustes dan lo que predice la teoria, pero dos numeros al principio no
cuadraban. En el experimento 1 la pendiente dio {p_fb_1:.2f} y esperabamos log2(3) = {np.log2(K_FIJO):.2f};
no era error de medicion sino que el costo real no es k^N sino (N+1)k^N, porque
evaluar una sola ruta tambien se encarece cuando N crece. Dividiendo por (N+1)
baja a {p_norm:.2f}.

El otro es la PD en el experimento 2, donde el cronometro dio {p_pd_2:.2f} en vez de 2
mientras las operaciones sobre los mismos grafos dan {o_pd_2:.2f}. Con N={N_FIJO} y k hasta {ks[-1]} la
PD hace menos de {ops_pd_2[-1]} operaciones, tan pocas que lo que se mide es el costo de
llamar a la funcion y no el algoritmo. Por eso se miden las dos cosas.

Entonces si, la PD es mas eficiente, y no por ser una version optimizada de la
fuerza bruta sino porque le cambia la clase de complejidad. Con N={Ns[-1]} y k={K_FIJO} ya hace
{ops_fb_1[-1] / ops_pd_1[-1]:,.0f} veces menos trabajo, y dandole un segundo a cada uno la fuerza bruta
alcanza N={n_fb_t[-1]:.0f} mientras la PD llega a N={n_pd_t[-1]:,.0f}.
""")
# %% Discusion escrita: comparacion con Dijkstra
"""Punto 1.4 - comparacion con Dijkstra

Dijkstra (1959) sirve para caminos minimos desde un nodo fuente en grafos con
pesos no negativos. Le asigna a cada nodo una etiqueta d[v] que es una cota del
costo real, y en cada paso saca de una cola de prioridad el nodo sin establecer
que tenga la etiqueta mas chica, lo da por definitivo y revisa sus aristas
salientes, actualizando d[v] = d[u] + c(u,v) cuando mejora. Funciona porque al
sacar el minimo esa etiqueta ya no puede bajar: cualquier otro camino hasta ese
nodo tendria que pasar por uno de etiqueta mayor, y como los pesos no son
negativos solo sumaria. Con monticulo binario cuesta O((|V|+|E|) log|V|).

Se parece a nuestra programacion dinamica mas de lo que uno pensaria. Los dos
salen del principio de optimalidad de Bellman y ninguno se pone a enumerar
caminos: en vez de eso calculan un valor para cada NODO, que es J_i(a) hacia T en
la PD y d[v] desde S en Dijkstra. Los dos revisan cada arista una sola vez y van
guardando de donde vinieron para armar la ruta al final. El ahorro esta ahi y es
el mismo en los dos casos, porque los nodos son N*k+2 y las rutas son k^N.

La diferencia de fondo es de donde sale el orden en que se procesan los nodos. La
PD ya lo sabe, se lo da la estructura por etapas del grafo. Dijkstra funciona con
cualquier grafo, incluso con ciclos, asi que tiene que ir averiguando en cada
paso cual es el siguiente nodo seguro, y eso es lo que cuesta la cola de
prioridad: de ahi sale el log. Las otras dos diferencias vienen de lo mismo.
Dijkstra necesita que los pesos no sean negativos porque su argumento es voraz,
y la PD sobre un DAG no lo necesita. Y Dijkstra da el costo de S a todos los
nodos, mientras la PD hacia atras da el de todos los nodos a T. En este grafo
gana la PD, Theta(N k^2) = Theta(|E|) contra O(N k^2 log(Nk)), y encima lo mejor
de Dijkstra, que es poder parar apenas se establece el destino, aqui no sirve de
nada porque T es el ultimo nodo en establecerse.

Referencias
[1] Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
    Numerische Mathematik, 1(1), 269-271.
[2] Bellman, R. (1957). Dynamic Programming. Princeton University Press, cap III.
[3] Cormen, Leiserson, Rivest & Stein (2009). Introduction to Algorithms, 3ra
    ed. MIT Press, secs. 24.2 y 24.3.
"""
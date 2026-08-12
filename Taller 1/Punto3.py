"""
Punto 3 - Optimizacion con restricciones: penalizacion
Analisis Numerico (CM0844) - Taller 1
Nombre: Maria Jose Gil Herrera

"""
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# BFGS + Armijo

def backtracking_armijo(f, grad_f, x, d, gx=None, gradx=None, c1=1e-4, rho=0.5, alpha0=1.0, max_back=60):
    # busca el primer alpha que cumpla armijo, empezando en 1 y bajando a la mitad
    if gx is None:
        gx = f(x)
    if gradx is None:
        gradx = grad_f(x)
    pendiente = gradx @ d
    alpha = alpha0
    for _ in range(max_back):
        if f(x + alpha * d) <= gx + c1 * alpha * pendiente:
            return alpha
        alpha *= rho
    return alpha  # si no encontro nada bueno, devuelve el ultimo que probo


def metodo_bfgs(f, grad_f, x0, tol=1e-8, max_iter=2000):
    x = np.array(x0, dtype=float)
    n = len(x)
    B_inv = np.eye(n)  # arranca aproximando el hessiano inverso como la identidad
    historia_norm = []
    gradx = grad_f(x)

    for k in range(max_iter):
        gx = f(x)
        norm_g = np.linalg.norm(gradx)
        historia_norm.append(norm_g)
        if norm_g < tol:
            break

        d = -B_inv @ gradx
        if gradx @ d >= 0:
            # si por algun motivo la direccion no es de descenso, nos devolvemos
            # al gradiente normal y reiniciamos la aproximacion
            d = -gradx
            B_inv = np.eye(n)

        alpha = backtracking_armijo(f, grad_f, x, d, gx, gradx)
        x_nuevo = x + alpha * d
        grad_nuevo = grad_f(x_nuevo)

        # actualizacion BFGS 
        s = (x_nuevo - x).reshape(-1, 1)
        y = (grad_nuevo - gradx).reshape(-1, 1)
        sy = float((s.T @ y).item())

        if sy > 1e-10:  # si no, mejor no actualizar (se puede dañar la aproximacion)
            rho_bfgs = 1.0 / sy
            I = np.eye(n)
            B_inv = (I - rho_bfgs * s @ y.T) @ B_inv @ (I - rho_bfgs * y @ s.T) + rho_bfgs * s @ s.T

        x = x_nuevo
        gradx = grad_nuevo

    return x, historia_norm, k + 1


def gradiente_numerico(f, x, h=1e-6):
    # diferencias finitas centradas
    x = np.array(x, dtype=float)
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

# Penalizacion exterior

def penalizacion_cuadratica(x, f, h_list, g_list, mu):
    val = f(x)
    for h in h_list:
        val += mu * h(x)**2
    for g in g_list:
        val += mu * max(g(x), 0.0)**2  # solo suma si g(x) > 0, o sea si se viola
    return val


def penalizacion_exterior(f, h_list, g_list, x0, mu1=1.0, eta=10.0, eps=1e-5,
                           max_outer=30, bfgs_tol=1e-8, verbose=True):
    # h_list y g_list son listas de funciones, para poder meter varias
    # restricciones sin tener que reescribir la funcion cada vez
    x = np.array(x0, dtype=float)
    mu = mu1
    trayectoria = []

    for n in range(1, max_outer + 1):
        # arma el subproblema sin restricciones para este mu, y lo resuelve con BFGS
        P = lambda xx, mu=mu: penalizacion_cuadratica(xx, f, h_list, g_list, mu)
        gradP = lambda xx, mu=mu: gradiente_numerico(P, xx)

        x_nuevo, hist_bfgs, n_iter_bfgs = metodo_bfgs(P, gradP, x, tol=bfgs_tol)

        cambio = np.linalg.norm(x_nuevo - x)
        viol_h = sum(h(x_nuevo)**2 for h in h_list)
        viol_g = sum(max(g(x_nuevo), 0.0)**2 for g in g_list)

        info = {"n": n, "mu": mu, "x": x_nuevo.copy(), "f": f(x_nuevo),
                "viol_h": viol_h, "viol_g": viol_g, "cambio": cambio,
                "iter_bfgs": n_iter_bfgs}
        trayectoria.append(info)

        if verbose:
            print(f"  n={n:>2}  mu={mu:>12.1f}   x=({x_nuevo[0]:.6f},{x_nuevo[1]:.6f})   "
                  f"f(x)={info['f']:.6f}   viol_h={viol_h:.2e}   viol_g={viol_g:.2e}   "
                  f"||x_n-x_(n-1)||={cambio:.2e}   (BFGS: {n_iter_bfgs} iter)")

        x = x_nuevo
        if cambio < eps and n > 1:
            if verbose:
                print(f"  -> listo, ya convergio (cambio={cambio:.2e} < eps={eps:.0e})")
            break

        mu *= eta  # subimos el castigo para la siguiente vuelta

    return x, trayectoria

print("verificacion del gradiente")

f = lambda x: x[0]**2 + x[1]**2
h1 = lambda x: x[0] + x[1] - 4
g1 = lambda x: x[0] - x[1] - 1

def gradP_analitico(x, mu):
    x1, x2 = x
    h = h1(x)
    gp = max(g1(x), 0.0)
    dP_dx1 = 2 * x1 + mu * (2 * h * 1 + 2 * gp * 1)
    dP_dx2 = 2 * x2 + mu * (2 * h * 1 + 2 * gp * (-1))
    return np.array([dP_dx1, dP_dx2])


# probamos en un punto cualquiera, para ver si numerico y analitico dan lo mismo
x_prueba = np.array([0.5, 0.3])
mu_prueba = 7.0
P_prueba = lambda xx: penalizacion_cuadratica(xx, f, [h1], [g1], mu_prueba)
grad_num = gradiente_numerico(P_prueba, x_prueba)
grad_ana = gradP_analitico(x_prueba, mu_prueba)
print(f"En x={x_prueba}, mu={mu_prueba}:")
print(f"  gradiente numerico  = {np.round(grad_num, 5)}")
print(f"  gradiente analitico = {grad_ana}")

# Resolviendo el problema de verdad

print("PUNTO 3 - Aplicando penalizacion exterior")

print("min x1^2+x2^2  s.a. x1+x2-4=0, x1-x2-1<=0,  x0=(0,0)\n")

x0 = np.array([0.0, 0.0])
x_final, tray = penalizacion_exterior(f, [h1], [g1], x0, mu1=1.0, eta=10.0, eps=1e-6)

x_exacto = np.array([2.0, 2.0])
f_exacto = 8.0
print(f"\nResultado final: x = {x_final}")
print(f"f(x) = {f(x_final):.6f}")
print(f"Solucion exacta (KKT): x* = (2,2), f* = 8")
print(f"Error final: ||x - x*|| = {np.linalg.norm(x_final - x_exacto):.2e}")

# Graficas

# Grafica donde va cayendo x_n mientras el metodo avanza.
fig, ax = plt.subplots(figsize=(7, 6))

xx = np.linspace(-0.5, 3.5, 300)
yy = np.linspace(-0.5, 3.5, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = XX**2 + YY**2
ax.contour(XX, YY, ZZ, levels=15, cmap="Greys", linewidths=0.6)

ax.plot(xx, 4 - xx, color="tab:blue", lw=2, label="h(x): x1+x2=4")

ax.fill_between(xx, xx - 1, 3.5, color="tab:orange", alpha=0.12,
                 label="region factible de g: x1-x2<=1")
ax.plot(xx, xx - 1, color="tab:orange", lw=1.3, ls="--")

xs = np.array([x0] + [info["x"] for info in tray])  # meto x0 al inicio para que la linea no quede cortada
ax.plot(xs[:, 0], xs[:, 1], "o-", color="tab:red", lw=2, ms=7, label="trayectoria x_n (penalizacion)")
ax.plot(0, 0, "s", color="purple", ms=9, label="inicio x0=(0,0)")
ax.plot(2, 2, "*", color="black", ms=20, label="optimo exacto (2,2)")

for info in tray:
    ax.annotate(f"n={info['n']}", info["x"], textcoords="offset points", xytext=(6, 4), fontsize=7)

ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 3.5)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Penalizacion exterior: trayectoria hacia el optimo restringido")
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
plt.show()
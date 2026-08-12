"""
Punto 2 - Optimizacion no lineal SIN restricciones
Analisis Numerico (CM0844) - Taller 1
Nombre: Maria Jose Gil Herrera

"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def f_rosenbrock(x):
    x1, x2 = x
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2


def grad_rosenbrock(x):
    x1, x2 = x
    df_dx1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])


def hess_rosenbrock(x):
    x1, x2 = x
    return np.array([
        [1200 * x1**2 - 400 * x2 + 2, -400 * x1],
        [-400 * x1, 200],
    ])


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


def hessiano_numerico(f, x, h=1e-4):
    # lo mismo pero de segundo orden
    x = np.array(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


print("verificacion del gradiente y el Hessiano (contra diferencias finitas)")

x_prueba = np.array([-1.2, 1.0])
g_analitico = grad_rosenbrock(x_prueba)
g_numerico = gradiente_numerico(f_rosenbrock, x_prueba)
H_analitico = hess_rosenbrock(x_prueba)
H_numerico = hessiano_numerico(f_rosenbrock, x_prueba)

print(f"En x0 = {x_prueba}:")
print(f"  grad analitico = {g_analitico}")
print(f"  grad numerico  = {np.round(g_numerico, 4)}")
print(f"  Hessiano analitico =\n{H_analitico}")
print(f"  Hessiano numerico  =\n{np.round(H_numerico, 2)}")

# Backtracking con Armijo 

def backtracking_armijo(f, grad_f, x, d, gx=None, gradx=None, c1=1e-4, rho=0.5, alpha0=1.0, max_back=60):

    if gx is None:
        gx = f(x)
    if gradx is None:
        gradx = grad_f(x)
    pendiente = gradx @ d  # tiene que dar < 0 (direccion de descenso)
    alpha = alpha0
    for _ in range(max_back):
        if f(x + alpha * d) <= gx + c1 * alpha * pendiente:
            return alpha
        alpha *= rho
    return alpha  # si no encontro nada, devuelve el mas chico que probo


# Metodo del gradiente

def metodo_gradiente(f, grad_f, x0, tol=1e-6, max_iter=50000):
    x = np.array(x0, dtype=float)
    historia_norm = []
    historia_x = [x.copy()]
    for k in range(max_iter):
        gx = f(x)
        gradx = grad_f(x)
        norm_g = np.linalg.norm(gradx)
        historia_norm.append(norm_g)
        if norm_g < tol:
            break
        d = -gradx  # direccion de maximo descenso
        alpha = backtracking_armijo(f, grad_f, x, d, gx, gradx)
        x = x + alpha * d
        historia_x.append(x.copy())
    return x, historia_norm, historia_x, k + 1


# Newton amortiguado (con backtracking)

def metodo_newton_amortiguado(f, grad_f, hess_f, x0, tol=1e-10, max_iter=1000):
    x = np.array(x0, dtype=float)
    historia_norm = []
    historia_x = [x.copy()]
    for k in range(max_iter):
        gx = f(x)
        gradx = grad_f(x)
        norm_g = np.linalg.norm(gradx)
        historia_norm.append(norm_g)
        if norm_g < tol:
            break

        H = hess_f(x)
        try:
            d = np.linalg.solve(H, -gradx)  # direccion de Newton: -H^-1 * grad
        except np.linalg.LinAlgError:
            d = -gradx  # si el Hessiano sale singular, mejor gradiente

        if gradx @ d >= 0:
            d = -gradx

        alpha = backtracking_armijo(f, grad_f, x, d, gx, gradx)  # "amortiguado" = alpha no fijo en 1
        x = x + alpha * d
        historia_x.append(x.copy())
    return x, historia_norm, historia_x, k + 1


# BFGS

def metodo_bfgs(f, grad_f, x0, tol=1e-10, max_iter=1000):
    x = np.array(x0, dtype=float)
    n = len(x)
    B_inv = np.eye(n)  # arranca aproximando el inverso del Hessiano como la identidad
    historia_norm = []
    historia_x = [x.copy()]
    gradx = grad_f(x)

    for k in range(max_iter):
        gx = f(x)
        norm_g = np.linalg.norm(gradx)
        historia_norm.append(norm_g)
        if norm_g < tol:
            break

        d = -B_inv @ gradx
        if gradx @ d >= 0:
            d = -gradx
            B_inv = np.eye(n)  # si algo salio mal, reinicia la aproximacion

        alpha = backtracking_armijo(f, grad_f, x, d, gx, gradx)
        x_nuevo = x + alpha * d
        grad_nuevo = grad_f(x_nuevo)

        s = (x_nuevo - x).reshape(-1, 1)         # cambio en x
        y = (grad_nuevo - gradx).reshape(-1, 1)  # cambio en el gradiente
        sy = float((s.T @ y).item())

        if sy > 1e-10:  # solo actualiza si mantiene curvatura positiva
            rho_bfgs = 1.0 / sy
            I = np.eye(n)
            B_inv = (I - rho_bfgs * s @ y.T) @ B_inv @ (I - rho_bfgs * y @ s.T) + rho_bfgs * s @ s.T

        x = x_nuevo
        gradx = grad_nuevo
        historia_x.append(x.copy())

    return x, historia_norm, historia_x, k + 1


# Corriendo los tres sobre Rosenbrock, desde x0=(-1.2,1)

print("aplicando los tres metodos a Rosenbrock")
print("x0 = (-1.2, 1),   optimo conocido x* = (1,1), f(x*) = 0\n")

def imprimir_progreso(nombre, historia_x, historia_norm, n_iter, n_checkpoints=8):
    # solo imprime unos pocos puntos espaciados para ver como va avanzando
    print(f"- {nombre}: progreso (mostrando {n_checkpoints} puntos de {n_iter} iteraciones)")
    idxs = np.unique(np.linspace(0, len(historia_x) - 1, n_checkpoints).astype(int))
    for i in idxs:
        x_i = historia_x[i]
        print(f"  iter {i:>6}:  x = {np.round(x_i, 5)}   ||grad|| = {historia_norm[i]:.3e}")
    print()


x0 = np.array([-1.2, 1.0])
x_estrella = np.array([1.0, 1.0])

x_g, hist_g, path_g, n_g = metodo_gradiente(f_rosenbrock, grad_rosenbrock, x0, tol=1e-6)
imprimir_progreso("Metodo del GRADIENTE", path_g, hist_g, n_g)

x_n, hist_n, path_n, n_n = metodo_newton_amortiguado(f_rosenbrock, grad_rosenbrock, hess_rosenbrock, x0)
imprimir_progreso("Metodo de NEWTON amortiguado", path_n, hist_n, n_n, n_checkpoints=min(8, n_n))

x_b, hist_b, path_b, n_b = metodo_bfgs(f_rosenbrock, grad_rosenbrock, x0)
imprimir_progreso("Metodo BFGS", path_b, hist_b, n_b, n_checkpoints=min(8, n_b))

print("resultado final de cada metodo:")
print(f"  Gradiente : x* = {x_g}   f(x*) = {f_rosenbrock(x_g):.3e}   iteraciones = {n_g}")
print(f"  Newton    : x* = {x_n}   f(x*) = {f_rosenbrock(x_n):.3e}   iteraciones = {n_n}")
print(f"  BFGS      : x* = {x_b}   f(x*) = {f_rosenbrock(x_b):.3e}   iteraciones = {n_b}")


# Graficas

# Grafica 1: zoom a las primeras iteraciones
fig, ax = plt.subplots(figsize=(7.5, 5))
zoom = 40
ax.semilogy(hist_g[:zoom], "o-", label="Gradiente", color="tab:red", ms=3)
ax.semilogy(hist_n[:zoom], "o-", label="Newton amortiguado", color="tab:blue", ms=4)
ax.semilogy(hist_b[:zoom], "o-", label="BFGS", color="tab:green", ms=4)
ax.set_xlabel("iteracion k")
ax.set_ylabel("||grad f(xk)||  (escala log)")
ax.set_title(f"Zoom: primeras {zoom} iteraciones (Newton y BFGS ya casi terminan aqui)")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
plt.show()

# Grafica 2: las trayectorias sobre las curvas de nivel de Rosenbrock
fig, ax = plt.subplots(figsize=(7.5, 6))
xx = np.linspace(-1.5, 1.5, 300)
yy = np.linspace(-0.5, 1.5, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = (1 - XX)**2 + 100 * (YY - XX**2)**2
ax.contour(XX, YY, ZZ, levels=np.logspace(-1, 3.5, 25), cmap="Greys", linewidths=0.6)

path_g_arr = np.array(path_g)
path_n_arr = np.array(path_n)
path_b_arr = np.array(path_b)
ax.plot(path_g_arr[:, 0], path_g_arr[:, 1], color="tab:red", lw=1.2, alpha=0.8, label=f"Gradiente ({n_g} iter)")
ax.plot(path_n_arr[:, 0], path_n_arr[:, 1], "o-", color="tab:blue", lw=1.5, ms=4, label=f"Newton ({n_n} iter)")
ax.plot(path_b_arr[:, 0], path_b_arr[:, 1], "o-", color="tab:green", lw=1.5, ms=4, label=f"BFGS ({n_b} iter)")
ax.plot(1, 1, "*", color="black", ms=18, label="Optimo (1,1)")
ax.plot(-1.2, 1, "s", color="purple", ms=8, label="Inicio (-1.2,1)")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Trayectorias sobre las curvas de nivel de Rosenbrock")
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# Grafica 3: la forma de comprobar el orden de convergencia. Si
# ||e_(k+1)|| ~ C*||e_k||^p, entonces log(e_(k+1)) = log(C) + p*log(e_k),
# osea que es una RECTA y su PENDIENTE es el orden p

def log_errores_consecutivos(path, x_estrella):
    errores = np.array([np.linalg.norm(np.array(xi) - x_estrella) for xi in path])
    # se descartan errores muy chicos (<1e-9)
    errores = errores[(errores > 1e-9) & (errores < 1e2)]
    return np.log10(errores[:-1]), np.log10(errores[1:])  # log(e_k), log(e_(k+1))


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

config = [
    ("Gradiente", path_g, "tab:red", 15, 1.0, "referencia pendiente 1 (lineal)"),
    ("Newton", path_n, "tab:blue", 5, 2.0, "referencia pendiente 2 (cuadratica)"),
    ("BFGS", path_b, "tab:green", 6, 1.5, "referencia pendiente 1.5 (a mitad de camino)"),
]
pendientes = {}

for ax, (nombre, path, color, n_ajuste, p_ref, label_ref) in zip(axes, config):
    lx, ly = log_errores_consecutivos(path, x_estrella)
    ax.scatter(lx, ly, s=22, color=color, alpha=0.55, label="datos")

    # recta ajustada a los ultimos puntos (zona ya asintotica) 
    lx_fit, ly_fit = lx[-n_ajuste:], ly[-n_ajuste:]
    p, log_C = np.polyfit(lx_fit, ly_fit, 1)
    pendientes[nombre] = p
    x_recta = np.linspace(lx.min(), lx.max(), 10)
    ax.plot(x_recta, p * x_recta + log_C, "-", color=color, lw=3,
             label=f"ajuste: pendiente = {p:.2f}")

    # recta de referencia con pendiente exxacta (1, 1.5 o 2), anclada al
    # mismo punto final para poder comparar a simple vista
    punto_ancla = (lx_fit[-1], ly_fit[-1])
    log_C_ref = punto_ancla[1] - p_ref * punto_ancla[0]
    ax.plot(x_recta, p_ref * x_recta + log_C_ref, ":", color="black", lw=2,
             label=label_ref)

    ax.set_xlabel("log10( ||e_k|| )")
    ax.set_ylabel("log10( ||e_(k+1)|| )")
    ax.set_title(f"{nombre}: pendiente empirica = {p:.2f}")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

fig.suptitle("Orden de convergencia", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94]) 
plt.show()
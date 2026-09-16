"""
Punto 2 - Regulador lineal-cuadratico: Riccati e integracion numerica
Analisis Numerico (CM0844) - Taller 2
Nombre: Maria Jose Gil Herrera
"""
# %% Modelo y ecuacion de Riccati
# ecuacion de Riccati, forma estandar (Kirk 2004 sec 5.2):
#   -P' = A'P + PA - P B R^-1 B' P + Q,   P(T) = 0,   u* = -R^-1 B' P x*
# el enunciado escribe 2Q en vez de Q, que equivale a peso 4I; se deja como
# bandera y la celda de la trayectoria verifica cual es consistente
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import solve_continuous_are
np.set_printoptions(suppress=True, precision=6)

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
Q = 2.0 * np.eye(2)
R = np.array([[1.0]])
R_INV = np.linalg.inv(R)

T_FINAL = 8.0
X0 = np.array([100.0, -20.0])      # 100 m de altura, bajando a 20 m/s

CONVENCION = "enunciado"          # "enunciado" -> +2Q   |   "estandar" -> +Q
Q_RICCATI = 2.0 * Q if CONVENCION == "enunciado" else Q

def riccati_derivada(P, Qr=None):
    Qr = Q_RICCATI if Qr is None else Qr
    return -(A.T @ P + P @ A - P @ B @ R_INV @ B.T @ P + Qr)

print(f"Modelo listo. T = {T_FINAL} s, x(0) = {X0}, convencion '{CONVENCION}'\n")
# %% Integrador propio por diferencias finitas
# de la diferencia P'(t) ~ [P(t) - P(t-dt)]/dt sale el paso retrogrado
# P(t-dt) = P(t) - dt P'(t): Euler explicito, de orden 1. Se integra hacia atras
# porque el dato conocido, P(T) = 0, esta al final

def integrar_riccati_propio(dt, Qr=None):
    n_pasos = int(round(T_FINAL / dt))
    dt = T_FINAL / n_pasos                  # para que la malla cierre exacta
    ts = np.linspace(0.0, T_FINAL, n_pasos + 1)
    Ps = np.zeros((n_pasos + 1, 2, 2))      # Ps[-1] = P(T) = 0

    for i in range(n_pasos, 0, -1):
        P = Ps[i]
        P_anterior = P - dt * riccati_derivada(P, Qr)
        Ps[i - 1] = 0.5 * (P_anterior + P_anterior.T)

    return ts, Ps

DT_PROPIO = 1e-3
ts_propio, Ps_propio = integrar_riccati_propio(DT_PROPIO)

print(f"Punto 2.1 - Euler explicito hacia atras, dt = {DT_PROPIO}\n")
print(f"  {'t [s]':>7} {'P11':>12} {'P12':>12} {'P22':>12}")
for t_m in [0.0, 2.0, 4.0, 6.0, 7.5, 8.0]:
    i = int(round(t_m / DT_PROPIO))
    P = Ps_propio[i]
    print(f"  {ts_propio[i]:>7.2f} {P[0, 0]:>12.6f} {P[0, 1]:>12.6f} {P[1, 1]:>12.6f}")

# lejos de T, P(t) debe estabilizarse en la solucion de la ecuacion ALGEBRAICA
# de Riccati, que scipy calcula por otro metodo
P_inf = solve_continuous_are(A, B, Q_RICCATI, R)
print(f"\n  P(0) obtenido      = {Ps_propio[0].ravel()}")
print(f"  P_inf (ARE, scipy) = {P_inf.ravel()}")
print(f"  ||P(0) - P_inf||_F = {np.linalg.norm(Ps_propio[0] - P_inf, 'fro'):.3e}")
print("  Coinciden: con T = 8 s el horizonte es largo comparado con la constante")
print("  de tiempo del lazo cerrado, asi que P ya llego al regimen estacionario.")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(ts_propio, Ps_propio[:, 0, 0], lw=1.8, label="$P_{11}$")
ax.plot(ts_propio, Ps_propio[:, 0, 1], lw=1.8, label="$P_{12} = P_{21}$")
ax.plot(ts_propio, Ps_propio[:, 1, 1], lw=1.8, label="$P_{22}$")
for v in (P_inf[0, 0], P_inf[0, 1], P_inf[1, 1]):
    ax.axhline(v, color="gray", ls="--", lw=0.9, alpha=0.7)
ax.plot([], [], color="gray", ls="--", lw=0.9, label="solucion estacionaria (ARE)")
ax.set(xlabel="t [s]", ylabel="entradas de P(t)",
       title=f"Punto 2.1: ecuacion de Riccati, dt = {DT_PROPIO}, P(T) = 0")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
print("\nMostrando: entradas de P(t) en [0, T]\n")
# %% Runge-Kutta de biblioteca y refinamiento de dt
# solve_ivp integra hacia atras con t_span = (T, 0); P se aplana en 4 entradas
# porque trabaja con vectores, y dense_output permite evaluarla en la misma malla

def riccati_vectorizada(t, p_plano):
    return riccati_derivada(p_plano.reshape(2, 2)).ravel()

sol_rk45 = solve_ivp(riccati_vectorizada, (T_FINAL, 0.0), np.zeros(4),
                     method="RK45", dense_output=True, rtol=1e-10, atol=1e-12)
P_rk45 = sol_rk45.sol(ts_propio).T.reshape(-1, 2, 2)

paso_rk45 = np.abs(np.diff(sol_rk45.t))
t_medio = 0.5 * (sol_rk45.t[:-1] + sol_rk45.t[1:])
error_en_t = np.linalg.norm(Ps_propio - P_rk45, axis=(1, 2))

print("Punto 2.2 - comparacion contra RK45 (rtol=1e-10, atol=1e-12)\n")
print(f"  RK45: {sol_rk45.t.size - 1} pasos, {sol_rk45.nfev} evaluaciones del lado derecho")
print(f"  propio: {len(ts_propio) - 1} pasos con dt = {DT_PROPIO} fijo")
# el primer paso de RK45 es un tanteo inicial, no refleja la adaptacion
print(f"  paso de RK45 (sin el tanteo inicial): min {paso_rk45[1:].min():.4f} s, "
      f"max {paso_rk45[1:].max():.4f} s, razon {paso_rk45[1:].max() / paso_rk45[1:].min():.0f}x")
print(f"\n  error ||P_propio - P_RK45||_F: {error_en_t[0]:.2e} en t=0, maximo "
      f"{error_en_t.max():.2e} en t = {ts_propio[error_en_t.argmax()]:.2f} s")

# Euler es de orden 1, asi que el error global deberia caer como C dt
pasos = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
errores_max, errores_0 = [], []

print(f"\n  {'dt':>10} {'pasos':>9} {'error max':>13} {'error en t=0':>14} {'razon':>7}")
for dt in pasos:
    ts_d, Ps_d = integrar_riccati_propio(dt)
    err = np.linalg.norm(Ps_d - sol_rk45.sol(ts_d).T.reshape(-1, 2, 2), axis=(1, 2))
    errores_max.append(err.max())
    errores_0.append(err[0])
    razon = errores_max[-2] / errores_max[-1] if len(errores_max) > 1 else np.nan
    print(f"  {dt:>10.4f} {int(T_FINAL / dt):>9,} {err.max():>13.3e} "
          f"{err[0]:>14.3e} {razon:>7.2f}")

# dt = 0.5 queda fuera del regimen asintotico (error ~ 1), se excluye del ajuste
pendiente, intercepto = np.polyfit(np.log(pasos[1:]), np.log(errores_max[1:]), 1)
res = np.log(errores_max[1:]) - (pendiente * np.log(pasos[1:]) + intercepto)
r2 = 1 - np.sum(res ** 2) / np.sum((np.log(errores_max[1:])
                                    - np.mean(np.log(errores_max[1:]))) ** 2)
print(f"\n  log(error) vs log(dt) sin dt=0.5: pendiente {pendiente:.4f}, R2 {r2:.5f} "
      f"(teoria 1)")

fig, (ax_e, ax_c, ax_h) = plt.subplots(1, 3, figsize=(15, 4.6))
ax_e.semilogy(ts_propio, error_en_t, lw=1.5, color="#c0392b")
ax_e.set(xlabel="t [s]", ylabel=r"$\|P_{propio}-P_{RK45}\|_F$",
         title=f"Error en [0, T] (dt = {DT_PROPIO})")
ax_c.loglog(pasos, errores_max, "o-", color="#2471a3", ms=6,
            label=f"error maximo ({pendiente:.3f})")
ax_c.loglog(pasos, errores_0, "s--", color="#148f77", ms=5, label="error en t = 0")
ax_c.loglog(pasos, np.array(pasos) * errores_max[-1] / pasos[-1], ":", color="gray",
            label="pendiente 1")
ax_c.set(xlabel=r"$\Delta t$", ylabel="error", title="Convergencia al refinar el paso")
ax_h.semilogy(t_medio[1:], paso_rk45[1:], ".-", color="#b7950b", lw=1.2, ms=5,
              label="paso de RK45")
ax_h.axhline(DT_PROPIO, color="#2471a3", ls="--", lw=1.4, label=f"dt fijo = {DT_PROPIO}")
ax_h.set(xlabel="t [s]", ylabel="tamano del paso [s]",
         title="RK45 ajusta el paso segun la solucion")
for a in (ax_e, ax_c, ax_h):
    a.grid(True, which="both", alpha=0.3)
for a in (ax_c, ax_h):
    a.legend(fontsize=8)
fig.suptitle("Punto 2.2: integrador propio vs RK45")
fig.tight_layout()
plt.show()

print(f"""
Mostrando: error en [0,T], convergencia al refinar dt, y paso de RK45

El error no queda parejo a lo largo del intervalo. En t = 0 vale {error_en_t[0]:.1e}, porque
ahi P ya es practicamente constante y Euler casi no se equivoca, y sube hasta
{error_en_t.max():.1e} cerca de t = {ts_propio[error_en_t.argmax()]:.1f} s, que es donde P cae de golpe hasta 0. Con paso fijo
uno gasta lo mismo en las dos zonas, aunque una sea facil y la otra no. RK45 se
da cuenta solo y va cambiando el paso, en este caso por un factor de {paso_rk45[1:].max() / paso_rk45[1:].min():.0f}x.

Al refinar dt el error cae con pendiente {pendiente:.3f}, que es el orden 1 que se esperaba
de Euler: si dt se parte por la mitad, el error tambien. Eso se paga caro, a
RK45 le bastaron {sol_rk45.t.size - 1} pasos para llegar a rtol = 1e-10 y el integrador propio con
{int(T_FINAL / pasos[-1]):,} pasos se quedo en {errores_max[-1]:.1e}.
""")
# %% Trayectoria optima en lazo cerrado
# se propaga x*' = Ax* + Bu* hacia adelante con u* = -R^-1 B' P(t) x*. P(t) se
# necesita en instantes arbitrarios, asi que se interpola linealmente

def P_interpolada(t):
    p11 = np.interp(t, ts_propio, Ps_propio[:, 0, 0])
    p12 = np.interp(t, ts_propio, Ps_propio[:, 0, 1])
    p22 = np.interp(t, ts_propio, Ps_propio[:, 1, 1])
    return np.array([[p11, p12], [p12, p22]])

def control_optimo(t, x):
    return float(-(R_INV @ B.T @ P_interpolada(t) @ x)[0])

def dinamica_lazo_cerrado(t, x):
    return A @ x + B.ravel() * control_optimo(t, x)

sol_estado = solve_ivp(dinamica_lazo_cerrado, (0.0, T_FINAL), X0, method="RK45",
                       dense_output=True, rtol=1e-10, atol=1e-12)
t_malla = np.linspace(0.0, T_FINAL, 2001)
x_malla = sol_estado.sol(t_malla)
u_malla = np.array([control_optimo(t, x_malla[:, i]) for i, t in enumerate(t_malla)])

print("Punto 2.3 - trayectoria optima\n")
print(f"  {'t [s]':>7} {'altura [m]':>13} {'velocidad [m/s]':>18} {'u [m/s^2]':>13}")
for t_m in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]:
    i = int(round(t_m / T_FINAL * (len(t_malla) - 1)))
    print(f"  {t_malla[i]:>7.2f} {x_malla[0, i]:>13.4f} {x_malla[1, i]:>18.4f} "
          f"{u_malla[i]:>13.4f}")

# el costo optimo debe valer x(0)' P(0) x(0); compararlo contra la integral
# sobre la trayectoria valida todo lo anterior y decide la convencion de Riccati
integrando_Q = np.einsum("ij,jk,ik->i", x_malla.T, Q, x_malla.T) + u_malla ** 2
integrando_2Q = np.einsum("ij,jk,ik->i", x_malla.T, 2 * Q, x_malla.T) + u_malla ** 2
J_con_Q = trapezoid(integrando_Q, t_malla)
J_con_2Q = trapezoid(integrando_2Q, t_malla)
J_teorico = float(X0 @ Ps_propio[0] @ X0)
err_Q = abs(J_con_Q - J_teorico) / J_teorico
err_2Q = abs(J_con_2Q - J_teorico) / J_teorico

print(f"\n  x(T) = ({x_malla[0, -1]:.4f} m, {x_malla[1, -1]:.4f} m/s)")
print(f"\n  identidad J* = x(0)' P(0) x(0):")
print(f"    x(0)' P(0) x(0)                = {J_teorico:12.2f}")
print(f"    int (x'Qx + u^2) dt,  Q  = 2I  = {J_con_Q:12.2f}   (error rel. {err_Q:.1e})")
print(f"    int (x'2Qx + u^2) dt, 2Q = 4I  = {J_con_2Q:12.2f}   (error rel. {err_2Q:.1e})")

fig, ejes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
ejes[0].plot(t_malla, x_malla[0], lw=1.8, color="#2471a3")
ejes[0].set(ylabel="$x_1^*$  altura [m]",
            title="Punto 2.3: trayectoria optima con $u^*=-R^{-1}B^TPx^*$")
ejes[1].plot(t_malla, x_malla[1], lw=1.8, color="#148f77")
ejes[1].set(ylabel="$x_2^*$  velocidad [m/s]")
ejes[2].plot(t_malla, u_malla, lw=1.8, color="#c0392b")
ejes[2].set(ylabel="$u^*$  aceleracion [m/s$^2$]", xlabel="t [s]")
for e in ejes:
    e.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    e.grid(alpha=0.3)
fig.tight_layout()
plt.show()

i_vmin = int(np.argmin(x_malla[1]))
if CONVENCION == "enunciado":
    nota = (f"Sobre la convencion: con el 2Q del enunciado la identidad se cumple contra la\n"
            f"integral de peso 2Q = 4I (error {err_2Q:.1e}, solo cuadratura) y no contra la del\n"
            f"Q = 2I del funcional de costo (error {err_Q:.1e}). La ecuacion como esta escrita\n"
            f"resuelve el problema con peso 4I, no el planteado; con CONVENCION = \"estandar\"\n"
            f"se invierte. El comportamiento es el mismo, solo cambia lo agresivo del control.")
else:
    nota = (f"Sobre la convencion: con la forma estandar (+Q) la identidad cuadra contra la\n"
            f"integral de Q = 2I (error {err_Q:.1e}) y no contra la de 2Q = 4I (error {err_2Q:.1e}).\n"
            f"Esta es la version que resuelve el problema tal como esta planteado.")

print(f"""
Mostrando: altura, velocidad y control en [0, T]

Lo primero que llama la atencion es que en t = 0 el control vale {u_malla[0]:.1f} m/s^2, o
sea que apunta hacia abajo: en vez de frenar la caida la esta acelerando. Mirando
u* = -[P12, P22] x, el termino de la altura aporta {-Ps_propio[0][0, 1] * X0[0]:.1f} y el de la velocidad
{-Ps_propio[0][1, 1] * X0[1]:+.1f}, asi que gana el de la altura. Tiene sentido: estar a 100 m del suelo lo
penaliza Q durante todo el trayecto, entonces le conviene bajar rapido primero.
Despues la velocidad toca {x_malla[1, i_vmin]:.1f} m/s en t = {t_malla[i_vmin]:.2f} s, el control cambia de signo y
frena, y el dron termina en x(T) = ({x_malla[0, -1]:.3f} m, {x_malla[1, -1]:.3f} m/s). Cerca de T el control
se va a 0 porque P(T) = 0, o sea que ya no queda nada que penalizar.

El control maximo es {np.abs(u_malla).max():.0f} m/s^2, unas {np.abs(u_malla).max() / 9.81:.1f} g, que ningun dron aguanta. El LQR no
pone limites a u, solo lo penaliza con R u^2.

{nota}
""")
# %% Discusion escrita: metodos de paso adaptativo

"""Punto 2.4 - metodos de paso adaptativo

Un metodo de paso adaptativo ajusta el tamano de paso h durante la integracion
en lugar de fijarlo al inicio: en cada iteracion estima el error local E_n y lo
compara con una tolerancia eps dada, aceptando el paso si E_n <= eps o
repitiendolo con h menor si no, y aprovechando que E ~ C h^(p+1) para predecir
el paso siguiente mediante h_new = sigma h (eps/E_n)^(1/(p+1)).

RK45 es adaptativo porque emplea un par encajado de metodos de orden 4 y 5 que
comparten las mismas etapas k_i y difieren solo en los pesos, de modo que la
diferencia entre ambas aproximaciones, E_n = |y_(n+1)^(5) - y_(n+1)^(4)|, estima
el error y permite regular h automaticamente con apenas 6 evaluaciones de f por
paso.

Su ventaja frente a RK4 es que este ultimo debe arrastrar por todo el intervalo
el paso fino que exige la region mas dificil, mientras RK45 lo agranda donde la
solucion es suave y alcanza la misma precision con muchas menos evaluaciones;
ademas el usuario especifica la tolerancia en vez de h, y ante un comportamiento
brusco imprevisto RK45 reduce el paso automaticamente mientras RK4 lo atraviesa
sin dar senal de error. Eso se ve en lo medido aqui: RK45 vario su paso por {paso_rk45[1:].max() / paso_rk45[1:].min():.0f}x
segun la zona y llego a rtol = 1e-10 con {sol_rk45.nfev} evaluaciones, mientras el integrador
propio, de paso fijo, se quedo en {errores_max[-1]:.0e} con {int(T_FINAL / pasos[-1]):,} pasos.

Referencias
[1] Dormand, J. R. & Prince, P. J. (1980). A family of embedded Runge-Kutta
    formulae. J. Comput. Appl. Math., 6(1), 19-26.
[2] Hairer, Norsett & Wanner (1993). Solving ODEs I: Nonstiff Problems, 2da ed.
    Springer, secs. II.4 y II.5.
"""
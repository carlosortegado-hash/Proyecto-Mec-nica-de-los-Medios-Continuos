import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

st.title("Simulador: Línea de traza – Trayectoria – Línea de humo")

# --- Inputs ---
vx_str = st.text_input("Expresión de v_x(x,y,t):", "y")
vy_str = st.text_input("Expresión de v_y(x,y,t):", "-x")
x0 = st.number_input("Posición inicial x₀:", value=1.0)
y0 = st.number_input("Posición inicial y₀:", value=0.0)

start_button = st.button("Iniciar simulación (10 s)")

# --- Definiciones simbólicas ---
x, y, t = sp.symbols('x y t')
vx = sp.sympify(vx_str)
vy = sp.sympify(vy_str)

vx_f = sp.lambdify((x, y, t), vx, "numpy")
vy_f = sp.lambdify((x, y, t), vy, "numpy")

def ode_system(tt, XY):
    X, Y = XY
    return [vx_f(X, Y, tt), vy_f(X, Y, tt)]


if start_button:

    # Contenedores gráficos
    cont1, cont2, cont3 = st.columns(3)
    g1 = cont1.empty()
    g2 = cont2.empty()
    g3 = cont3.empty()

    # Expresiones debajo
    e1 = cont1.empty()
    e2 = cont2.empty()
    e3 = cont3.empty()

    # --- 1. Línea de TRAZA (streamline) ---
    streamline_ode = sp.simplify(vy.subs(t, 0) / vx.subs(t, 0))
    streamline_expr = sp.dsolve(sp.Eq(sp.diff(sp.Function('y')(x), x), streamline_ode))

    # --- Simulación durante 10 s ---
    TMAX = 10
    NFRAMES = 100
    times = np.linspace(0, TMAX, NFRAMES)

    # Historial de trayectoria
    X_traj = []
    Y_traj = []

    # Línea de humo: puntos que pasaron por (x0, y0)
    streak_x = []
    streak_y = []
    emission_times = np.linspace(-5, 0, 15)

    for ti in times:

        # 1) Trayectoria (pathline)
        sol = solve_ivp(ode_system, [0, ti], [x0, y0], max_step=0.05)
        Xp = sol.y[0]
        Yp = sol.y[1]

        # Guardar última posición
        X_traj.append(Xp[-1])
        Y_traj.append(Yp[-1])

        # 2) Línea de humo (streakline)
        sx = []
        sy = []
        for τ in emission_times:
            sol_s = solve_ivp(ode_system, [τ, ti], [x0, y0], max_step=0.05)
            sx.append(sol_s.y[0][-1])
            sy.append(sol_s.y[1][-1])
        streak_x = sx
        streak_y = sy

        # 3) Línea de traza instantánea
        streamline_fun = sp.lambdify((x, y), streamline_ode, "numpy")
        def ode_trace(s, Y):
            Xs, Ys = Y
            return [1, streamline_fun(Xs, Ys)]
        sol_trace = solve_ivp(ode_trace, [0, 6], [x0, y0], max_step=0.05)
        Xt = sol_trace.y[0]
        Yt = sol_trace.y[1]

        # ----- Gráficas -----

        # Línea de traza
        fig1, ax1 = plt.subplots()
        ax1.plot(Xt, Yt)
        ax1.set_title("Línea de traza")
        ax1.set_aspect("equal")
        g1.pyplot(fig1)

        # Trayectoria
        fig2, ax2 = plt.subplots()
        ax2.plot(Xp, Yp)
        ax2.set_title("Trayectoria")
        ax2.set_aspect("equal")
        g2.pyplot(fig2)

        # Línea de humo
        fig3, ax3 = plt.subplots()
        ax3.scatter(streak_x, streak_y)
        ax3.set_title("Línea de humo")
        ax3.set_aspect("equal")
        g3.pyplot(fig3)

        # Expresiones simbólicas
        e1.latex("Línea de traza:\\quad " + sp.latex(streamline_expr))
        e2.latex(r"Trayectoria:\quad (x(t),y(t))\ \text{calculada numéricamente}")
        e3.latex(r"Línea\ de\ humo:\quad \{x(t;\tau),y(t;\tau)\}")

        time.sleep(0.1)

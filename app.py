import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.title("Cálculo de líneas de traza, trayectorias y líneas de humo")

# --- Inputs del usuario ---
vx_str = st.text_input("Expresión de v_x(x,y,t):", "y")
vy_str = st.text_input("Expresión de v_y(x,y,t):", "-x")
t0 = st.number_input("Tiempo actual t₀:", value=0.0)
x0 = st.number_input("Posición inicial x₀:", value=1.0)
y0 = st.number_input("Posición inicial y₀:", value=0.0)

if st.button("Calcular"):

    # --- Definir variables simbólicas ---
    x, y, t = sp.symbols('x y t')
    vx = sp.sympify(vx_str)
    vy = sp.sympify(vy_str)

    # ---------- 1. LÍNEA DE TRAZA ----------
    st.subheader("Línea de traza (streamline)")

    # Resolver dy/dx = v_y / v_x evaluado en t=t0
    streamline_ode = sp.simplify(vy.subs(t, t0) / vx.subs(t, t0))

    st.write("Ecuación diferencial:")
    st.latex(r"\frac{dy}{dx} = " + sp.latex(streamline_ode))

    y_stream = sp.dsolve(sp.Eq(sp.diff(sp.Function('y')(x), x), streamline_ode))
    st.write("Solución general:")
    st.latex(sp.latex(y_stream))

    # Gráfica de la línea de traza
    xf = np.linspace(x0 - 2, x0 + 2, 400)
    # Convertir ODE a función numérica
    f_stream = sp.lambdify((x, y), streamline_ode.subs(t, t0), 'numpy')

    def integrate_streamline(x0, y0):
        def f(s, Y):
            X, Yv = Y
            return [1, f_stream(X, Yv)]
        sol = solve_ivp(f, [0, 6], [x0, y0], dense_output=True)
        X = sol.y[0]
        Y = sol.y[1]
        return X, Y

    Xs, Ys = integrate_streamline(x0, y0)

    fig, ax = plt.subplots()
    ax.plot(Xs, Ys)
    ax.set_title("Línea de traza")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    st.pyplot(fig)

    # ---------- 2. TRAYECTORIA ----------
    st.subheader("Trayectoria (pathline)")

    # Convertir velocidades a funciones numéricas
    vx_f = sp.lambdify((x, y, t), vx, 'numpy')
    vy_f = sp.lambdify((x, y, t), vy, 'numpy')

    def ode_path(tvar, XY):
        X, Y = XY
        return [vx_f(X, Y, tvar), vy_f(X, Y, tvar)]

    sol_path = solve_ivp(ode_path, [t0, t0 + 10], [x0, y0], dense_output=True)
    Xp = sol_path.y[0]
    Yp = sol_path.y[1]

    fig2, ax2 = plt.subplots()
    ax2.plot(Xp, Yp)
    ax2.set_title("Trayectoria")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    st.pyplot(fig2)

    st.write("Ecuaciones paramétricas aproximadas:")
    st.latex(f"x(t) ≈ x₀ + … (calculada numéricamente)")
    st.latex(f"y(t) ≈ y₀ + … (calculada numéricamente)")

    # ---------- 3. LÍNEA DE HUMO ----------
    st.subheader("Línea de humo (streakline)")

    # tiempos pasados para generar partículas
    times = np.linspace(t0 - 10, t0, 20)
    streak_x = []
    streak_y = []

    for τ in times:
        sol = solve_ivp(ode_path, [τ, t0], [x0, y0])
        streak_x.append(sol.y[0][-1])
        streak_y.append(sol.y[1][-1])

    fig3, ax3 = plt.subplots()
    ax3.scatter(streak_x, streak_y)
    ax3.set_title("Línea de humo")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    st.pyplot(fig3)

    st.success("Cálculo completado.")

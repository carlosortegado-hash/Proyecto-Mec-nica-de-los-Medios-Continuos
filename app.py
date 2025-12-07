import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="Cinemática de un fluido — Visualización")

st.title("Simulador visual de campos de velocidad, deformación y vorticidad")

st.markdown("""
Este programa visualiza:
- Campo vectorial del flujo
- Gradiente de velocidades \\(\\nabla v\\)
- Tensor de deformación unitaria \\(D\\)
- Tensor de rotación \\(W\\)
- Vorticidad \\(\\omega\\)
- Aceleración material \\(a = (v\\cdot\\nabla)v\\)
- Deformación real de una malla de partículas en el tiempo
""")

# ------------------------
# ENTRADAS DEL USUARIO
# ------------------------
st.sidebar.header("Campo de velocidades")

vx_str = st.sidebar.text_input("v_x(x,y) =", "y")
vy_str = st.sidebar.text_input("v_y(x,y) =", "-x")

xmin = st.sidebar.number_input("x min", -3.0)
xmax = st.sidebar.number_input("x max", 3.0)
ymin = st.sidebar.number_input("y min", -3.0)
ymax = st.sidebar.number_input("y max", 3.0)

t_sim = st.sidebar.number_input("Tiempo total de animación", 5.0)
fps = st.sidebar.slider("FPS", 5, 30, 12)

start_button = st.sidebar.button("Iniciar simulación")

# ------------------------
# SIMBOLIC DEFINITIONS
# ------------------------
x, y = sp.symbols('x y')
vx_expr = sp.sympify(vx_str)
vy_expr = sp.sympify(vy_str)

vx = sp.lambdify((x, y), vx_expr, "numpy")
vy = sp.lambdify((x, y), vy_expr, "numpy")

# symbolic gradient
dvx_dx = sp.diff(vx_expr, x)
dvx_dy = sp.diff(vx_expr, y)
dvy_dx = sp.diff(vy_expr, x)
dvy_dy = sp.diff(vy_expr, y)

# tensors symbolic
Dv = sp.Matrix([[dvx_dx, dvx_dy],
                [dvy_dx, dvy_dy]])

D_tensor = (Dv + Dv.T)/2
W_tensor = (Dv - Dv.T)/2
vorticity = dvy_dx - dvx_dy

# acceleration material a = (v·∇)v
ax_expr = vx_expr*dvx_dx + vy_expr*dvx_dy
ay_expr = vx_expr*dvy_dx + vy_expr*dvy_dy
a = sp.Matrix([ax_expr, ay_expr])

# ------------------------
# DISPLAY SYMBOLIC RESULTS
# ------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Gradiente de velocidades")
    st.latex(sp.latex(Dv))

    st.subheader("Tensor de deformación unitaria D")
    st.latex(sp.latex(D_tensor))

    st.subheader("Tensor de rotación W")
    st.latex(sp.latex(W_tensor))

with colB:
    st.subheader("Vorticidad")
    st.latex(r"\omega = " + sp.latex(vorticity))

    st.subheader("Aceleración material a")
    st.latex(sp.latex(a))

# ------------------------
# ANIMATION
# ------------------------
if start_button:
    st.markdown("---")
    st.subheader("Animación del campo de velocidades y deformación de una malla")

    anim_placeholder = st.empty()

    # grid for vector field
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 15),
                       np.linspace(ymin, ymax, 15))

    # initial grid of fluid particles
    grid_res = 9
    gx, gy = np.meshgrid(np.linspace(xmin+0.5, xmax-0.5, grid_res),
                         np.linspace(ymin+0.5, ymax-0.5, grid_res))

    # flatten positions for integration
    particles = np.vstack([gx.flatten(), gy.flatten()]).T

    dt = 1/fps
    steps = int(t_sim * fps)

    for _ in range(steps):
        # update particles with explicit Euler
        u = vx(particles[:,0], particles[:,1])
        v = vy(particles[:,0], particles[:,1])

        particles[:,0] += u * dt
        particles[:,1] += v * dt

        # DRAW FIGURE
        fig, ax = plt.subplots(figsize=(6,6))
        ax.quiver(X, Y, vx(X,Y), vy(X,Y), color="blue", alpha=0.5)
        ax.scatter(particles[:,0], particles[:,1], color="red", s=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title("Evolución de partículas bajo el campo de velocidades")

        anim_placeholder.pyplot(fig)
        time.sleep(dt)

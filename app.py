import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide", page_title="Simulación cinemática 3D — Medios Continuos")

st.title("Visualizador 3D (simplificado) de cinemática de un fluido")

st.markdown("""
### Este programa:
- Acepta campos de velocidad **3D dependientes de x, y, z y t**
- Calcula:  
  • Gradiente de velocidades \\(\\nabla v\\)  
  • Tensor de deformación unitaria \\(D\\)  
  • Vorticidad \\(\\boldsymbol{\\omega}\\)  
  • Aceleración material \\(a = \\partial_t v + (v\\cdot\\nabla)v\\)  
- Muestra una **animación 2D en el plano z = 0** con malla deformándose  
- Incluye **leyenda** indicando elementos de la gráfica
""")

# -----------------------------------------------------
# ENTRADAS DEL USUARIO
# -----------------------------------------------------
st.sidebar.header("Campo de velocidades")

vx_str = st.sidebar.text_input("v_x(x,y,z,t) =", "y")
vy_str = st.sidebar.text_input("v_y(x,y,z,t) =", "-x")
vz_str = st.sidebar.text_input("v_z(x,y,z,t) =", "0")

xmin = st.sidebar.number_input("x min", -3.0)
xmax = st.sidebar.number_input("x max", 3.0)
ymin = st.sidebar.number_input("y min", -3.0)
ymax = st.sidebar.number_input("y max", 3.0)

t_sim = st.sidebar.number_input("Tiempo total de animación", 5.0)
fps = st.sidebar.slider("FPS", 5, 30, 12)

start_button = st.sidebar.button("Iniciar simulación")

# -----------------------------------------------------
# DEFINICIONES SIMBÓLICAS
# -----------------------------------------------------
x, y, z, t = sp.symbols("x y z t")

vx_expr = sp.sympify(vx_str)
vy_expr = sp.sympify(vy_str)
vz_expr = sp.sympify(vz_str)

# Lambdify para animación (en z = 0)
vx = sp.lambdify((x, y, t), vx_expr.subs(z, 0), "numpy")
vy = sp.lambdify((x, y, t), vy_expr.subs(z, 0), "numpy")

# -----------------------------------------------------
# GRADIENTE DE VELOCIDADES
# -----------------------------------------------------
Dv = sp.Matrix([
    [sp.diff(vx_expr, x), sp.diff(vx_expr, y), sp.diff(vx_expr, z)],
    [sp.diff(vy_expr, x), sp.diff(vy_expr, y), sp.diff(vy_expr, z)],
    [sp.diff(vz_expr, x), sp.diff(vz_expr, y), sp.diff(vz_expr, z)]
])

# Tensor de deformación unitaria
D_tensor = (Dv + Dv.T) / 2

# Vorticidad
omega = sp.Matrix([
    sp.diff(vz_expr, y) - sp.diff(vy_expr, z),
    sp.diff(vx_expr, z) - sp.diff(vz_expr, x),
    sp.diff(vy_expr, x) - sp.diff(vx_expr, y)
])

# Aceleración material a = ∂v/∂t + (v·∇)v
dv_dt = sp.Matrix([sp.diff(vx_expr, t),
                   sp.diff(vy_expr, t),
                   sp.diff(vz_expr, t)])

adv = sp.Matrix([
    vx_expr*sp.diff(vx_expr, x) + vy_expr*sp.diff(vx_expr, y) + vz_expr*sp.diff(vx_expr, z),
    vx_expr*sp.diff(vy_expr, x) + vy_expr*sp.diff(vy_expr, y) + vz_expr*sp.diff(vy_expr, z),
    vx_expr*sp.diff(vz_expr, x) + vy_expr*sp.diff(vz_expr, y) + vz_expr*sp.diff(vz_expr, z)
])

a = dv_dt + adv

# -----------------------------------------------------
# MOSTRAR RESULTADOS SIMBÓLICOS
# -----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gradiente de velocidades")
    st.latex(sp.latex(Dv))

    st.subheader("Tensor de deformación unitaria D")
    st.latex(sp.latex(D_tensor))

with col2:
    st.subheader("Vorticidad ω")
    st.latex(sp.latex(omega))

    st.subheader("Aceleración material a")
    st.latex(sp.latex(a))

# -----------------------------------------------------
# ANIMACIÓN 2D (z = 0)
# -----------------------------------------------------
if start_button:
    st.markdown("---")
    st.subheader("Animación en el plano z = 0")

    anim_placeholder = st.empty()

    # Grid para el campo vectorial
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 15),
                       np.linspace(ymin, ymax, 15))

    # Malla de partículas
    gX, gY = np.meshgrid(np.linspace(xmin+0.5, xmax-0.5, 10),
                         np.linspace(ymin+0.5, ymax-0.5, 10))
    particles = np.vstack([gX.flatten(), gY.flatten()]).T

    dt = 1/fps
    steps = int(t_sim * fps)

    for step in range(steps):
        tt = step * dt

        # Update particle positions (Euler explícito)
        u = vx(particles[:,0], particles[:,1], tt)
        v = vy(particles[:,0], particles[:,1], tt)

        particles[:,0] += u * dt
        particles[:,1] += v * dt

        # FIGURA
        fig, ax = plt.subplots(figsize=(6,6))

        # Campo vectorial
        ax.quiver(X, Y, vx(X,Y,tt), vy(X,Y,tt), color="blue", alpha=0.5, label="Campo de velocidades")

        # Partículas
        ax.scatter(particles[:,0], particles[:,1], color="red", s=12, label="Partículas fluidas")

        # Límites
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        # Leyenda
        ax.legend(loc="upper right")

        ax.set_title(f"Evolución de partículas en el plano z = 0  |  t = {tt:.2f} s")

        anim_placeholder.pyplot(fig)
        time.sleep(dt)

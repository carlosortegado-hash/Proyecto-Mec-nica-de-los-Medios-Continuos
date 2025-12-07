import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import matplotlib.animation as animation

# Configuración de la página
st.set_page_config(page_title="Simulador de Fluidos", layout="wide")

def main():
    st.title("🌊 Simulador de Mecánica de Fluidos")
    st.markdown("Introduce las componentes del vector velocidad $\\vec{V} = (u, v, w)$.")

    # --- BARRA LATERAL (INPUTS) ---
    st.sidebar.header("Configuración del Campo")
    u_str = st.sidebar.text_input("Velocidad en X (u):", "2*x")
    v_str = st.sidebar.text_input("Velocidad en Y (v)", "-y")
    w_str = st.sidebar.text_input("Velocidad en Z (w)", "0.1*t")
    
    # Variables simbólicas
    x, y, z, t = sp.symbols('x y z t')

    if st.sidebar.button("Analizar y Simular"):
        try:
            u = sp.sympify(u_str)
            v = sp.sympify(v_str)
            w = sp.sympify(w_str)
        except:
            st.error("Error en las fórmulas. Revisa la sintaxis.")
            return

        # --- ANÁLISIS TEÓRICO ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Propiedades del Flujo")
            
            # 1. Estacionario
            derivadas_t = [sp.diff(c, t) for c in [u,v,w]]
            if all(d == 0 for d in derivadas_t):
                st.success("**Estacionario**: No depende del tiempo.")
            else:
                st.warning("**No Estacionario**: Depende del tiempo.")
                
            # 2. Dimensionalidad
            vars_p = set().union(*[c.free_symbols for c in [u,v,w]])
            coords = {x, y, z}
            dims = len(coords.intersection(vars_p))
            if w==0 and z not in vars_p: dims = 2
            st.info(f"**Dimensionalidad**: Flujo {dims}D")

        with col2:
            st.subheader("🧮 Operadores Diferenciales")
            
            # 3. Divergencia
            div = sp.simplify(sp.diff(u, x) + sp.diff(v, y) + sp.diff(w, z))
            st.write("Divergencia ($\\nabla \\cdot \\vec{V}$):")
            st.latex(sp.latex(div))
            if div == 0: st.caption("👉 Incompresible (Líquido)")
            else: st.caption("👉 Compresible (Gas)")
            
            # 4. Rotacional
            rot_x = sp.simplify(sp.diff(w, y) - sp.diff(v, z))
            rot_y = sp.simplify(sp.diff(u, z) - sp.diff(w, x))
            rot_z = sp.simplify(sp.diff(v, x) - sp.diff(u, y))
            st.write("Rotacional ($\\nabla \\times \\vec{V}$):")
            st.latex(f"({sp.latex(rot_x)})\\hat{{i}} + ({sp.latex(rot_y)})\\hat{{j}} + ({sp.latex(rot_z)})\\hat{{k}}")

        # --- SIMULACIÓN VISUAL ---
        st.markdown("---")
        st.subheader("🖥️ Simulación Visual")
        tipo = st.radio("Selecciona visualización:", ["Líneas de Corriente (Streamlines)", "Trayectorias (Pathlines)"], horizontal=True)

        with st.spinner("Generando animación (esto puede tardar unos segundos)..."):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

            # Funciones numéricas
            f_u = sp.lambdify((t, x, y, z), u, 'numpy')
            f_v = sp.lambdify((t, x, y, z), v, 'numpy')
            f_w = sp.lambdify((t, x, y, z), w, 'numpy')

            def get_vel(ti, pos):
                px, py, pz = pos
                try: 
                    # Manejo seguro de floats y arrays
                    vx = float(f_u(ti, px, py, pz))
                    vy = float(f_v(ti, px, py, pz))
                    vz = float(f_w(ti, px, py, pz))
                except: 
                    return np.array([0.0, 0.0, 0.0])
                return np.array([vx, vy, vz])

            if "Corriente" in tipo:
                # Streamlines
                semillas = [[i, j, 0] for i in range(-3, 4, 3) for j in range(-3, 4, 3)]
                lines = [ax.plot([], [], [], color='blue', alpha=0.6)[0] for _ in semillas]
                
                def update_stream(frame):
                    time_fix = frame * 0.1
                    ax.set_title(f"Líneas de Corriente (t={time_fix:.1f})")
                    for i, seed in enumerate(semillas):
                        camino = [seed]
                        curr = np.array(seed, dtype=float)
                        for _ in range(20): # Longitud de la linea
                            v = get_vel(time_fix, curr)
                            curr = curr + v * 0.2
                            camino.append(curr)
                        dat = np.array(camino)
                        lines[i].set_data(dat[:,0], dat[:,1])
                        lines[i].set_3d_properties(dat[:,2])
                    return lines

                ani = FuncAnimation(fig, update_stream, frames=30, interval=100)
            
            else:
                # Pathlines
                puntos = [[1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0]]
                lines = [ax.plot([], [], [], 'o-', markersize=4)[0] for _ in puntos]
                paths = [[np.array(p)] for p in puntos]
                
                def update_path(frame):
                    dt = 0.1
                    ti = frame * dt
                    ax.set_title(f"Trayectorias (t={ti:.1f})")
                    for i, line in enumerate(lines):
                        last = paths[i][-1]
                        v = get_vel(ti, last)
                        new_p = last + v * dt
                        paths[i].append(new_p)
                        dat = np.array(paths[i])
                        line.set_data(dat[:,0], dat[:,1])
                        line.set_3d_properties(dat[:,2])
                    return lines

                ani = FuncAnimation(fig, update_path, frames=40, interval=100)

            # Renderizar animación en Streamlit usando JSHTML
            st.components.v1.html(ani.to_jshtml(), height=600)

if __name__ == "__main__":
    main()

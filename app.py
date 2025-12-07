import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Configuración de la página
st.set_page_config(page_title="Calculadora de Fluidos 2D", layout="wide")

def main():
    st.title("🌊 Calculadora de Fluidos 2D (Analítica y Gráfica)")
    st.markdown("""
    Introduce las velocidades $u$ y $v$. Pueden depender de **x**, **y**, y el tiempo **t**.
    El programa intentará buscar la ecuación matemática y generará la gráfica.
    """)

    # --- 1. CONFIGURACIÓN LATERAL ---
    st.sidebar.header("1. Definir Campo de Velocidad")
    # Valores por defecto que no dan error
    u_input = st.sidebar.text_input("Velocidad en X (u):", "1 + 0.5*t")
    v_input = st.sidebar.text_input("Velocidad en Y (v):", "x")
    
    st.sidebar.header("2. Parámetros de Simulación")
    t_val = st.sidebar.number_input("Instante de tiempo (t) para visualizar:", value=2.0, step=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.write("Condiciones iniciales (para Trayectoria/Humo):")
    x0 = st.sidebar.number_input("Posición X0:", value=0.0)
    y0 = st.sidebar.number_input("Posición Y0:", value=0.0)

    # --- 2. PROCESAMIENTO MATEMÁTICO ---
    # Definimos símbolos
    x, y, t = sp.symbols('x y t')
    
    # Convertimos texto a funciones sympy
    try:
        u_sym = sp.sympify(u_input)
        v_sym = sp.sympify(v_input)
    except:
        st.error("Error en las fórmulas. Usa sintaxis Python: 2*x, sin(t), etc.")
        return

    # Funciones numéricas para las gráficas (lambdify)
    # 'numpy' permite que si metes arrays, salgan arrays
    func_u = sp.lambdify((t, x, y), u_sym, modules='numpy')
    func_v = sp.lambdify((t, x, y), v_sym, modules='numpy')

    # Función auxiliar para obtener velocidad en un punto (para trayectoria)
    def get_vel(time, pos):
        px, py = pos
        # Forzamos float para evitar errores con sympy numbers
        val_u = func_u(time, px, py)
        val_v = func_v(time, px, py)
        return [float(val_u), float(val_v)]

    # --- 3. MOSTRAR RESULTADOS ---
    
    st.subheader(f"Analizando el instante t = {t_val}")

    # Pestañas para organizar la info
    tab1, tab2, tab3 = st.tabs(["Líneas de Corriente", "Trayectoria", "Línea de Humo"])

    # ==========================================
    # PESTAÑA 1: LÍNEAS DE CORRIENTE (Streamlines)
    # ==========================================
    with tab1:
        col_math, col_graph = st.columns([1, 2])
        
        with col_math:
            st.markdown("### 📐 Ecuación")
            st.info("Se obtiene resolviendo: $\\frac{dy}{dx} = \\frac{v}{u}$ (t fijo)")
            
            try:
                u_fixed = u_sym.subs(t, t_val)
                v_fixed = v_sym.subs(t, t_val)
                equation = sp.Eq(sp.Derivative(y, x), v_fixed / u_fixed)
                st.latex(sp.latex(equation))
            except:
                st.warning("Ecuación compleja.")

        with col_graph:
            st.markdown("### 📈 Gráfica del Campo")
            fig, ax = plt.subplots()
            
            # Grid para streamplot
            Y, X = np.mgrid[-5:5:100j, -5:5:100j]
            
            # Calculamos U y V
            U_num = func_u(t_val, X, Y)
            V_num = func_v(t_val, X, Y)
            
            # --- CORRECCIÓN DEL ERROR ---
            # Si la velocidad es constante (ej: u=5), numpy devuelve un escalar.
            # Matplotlib necesita una matriz del mismo tamaño que X.
            # Lo arreglamos "transmitiendo" (broadcasting) el valor.
            if np.isscalar(U_num):
                U_num = np.full_like(X, U_num)
            if np.isscalar(V_num):
                V_num = np.full_like(X, V_num)
            # ----------------------------

            strm = ax.streamplot(X, Y, U_num, V_num, color=np.sqrt(U_num**2 + V_num**2), linewidth=1, cmap='autumn')
            ax.set_title(f"Líneas de Corriente en t={t_val}")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_xlim(-5,5); ax.set_ylim(-5,5)
            fig.colorbar(strm.lines, label="Velocidad")
            st.pyplot(fig)

    # ==========================================
    # PESTAÑA 2: TRAYECTORIA (Pathlines)
    # ==========================================
    with tab2:
        col_math, col_graph = st.columns([1, 2])
        
        with col_math:
            st.markdown("### 📐 Ecuación")
            st.write("Sistema:")
            st.latex(f"\\frac{{dx}}{{dt}} = {sp.latex(u_sym)}")
            st.latex(f"\\frac{{dy}}{{dt}} = {sp.latex(v_sym)}")
            
        with col_graph:
            st.markdown("### 📈 Gráfica de la Partícula")
            t_span = np.linspace(0, t_val, 100)
            
            def model_trayectoria(pos, time_var):
                return get_vel(time_var, pos)
            
            try:
                path = odeint(model_trayectoria, [x0, y0], t_span)
                fig2, ax2 = plt.subplots()
                ax2.plot(path[:,0], path[:,1], 'b-', label='Recorrido')
                ax2.plot(path[-1,0], path[-1,1], 'ro', label='Posición final')
                ax2.plot(x0, y0, 'go', label='Inicio')
                ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Error calculando trayectoria: {e}")

    # ==========================================
    # PESTAÑA 3: LÍNEA DE HUMO (Streaklines)
    # ==========================================
    with tab3:
        st.markdown("### 📈 Gráfica de Humo")
        try:
            taus = np.linspace(0, t_val, 40)
            humo_x = []
            humo_y = []
            
            for tau in taus:
                if t_val > tau:
                    t_vida = np.linspace(tau, t_val, 20)
                    trayectoria_p = odeint(model_trayectoria, [x0, y0], t_vida)
                    humo_x.append(trayectoria_p[-1][0])
                    humo_y.append(trayectoria_p[-1][1])
                else:
                    humo_x.append(x0)
                    humo_y.append(y0)

            fig3, ax3 = plt.subplots()
            ax3.plot(humo_x, humo_y, 'o-', color='purple', markersize=4, alpha=0.7)
            ax3.plot(x0, y0, 'r^', label='Inyector')
            ax3.set_xlim(-5, 5); ax3.set_ylim(-5, 5)
            ax3.grid(True)
            ax3.legend()
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Error calculando humo: {e}")

if __name__ == "__main__":
    main()

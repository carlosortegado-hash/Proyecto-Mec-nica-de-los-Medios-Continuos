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
    func_u = sp.lambdify((t, x, y), u_sym, modules=['numpy'])
    func_v = sp.lambdify((t, x, y), v_sym, modules=['numpy'])

    # Función auxiliar para obtener velocidad numérica
    def get_vel(time, pos):
        px, py = pos
        # Arrays de numpy protection
        return [float(func_u(time, px, py)), float(func_v(time, px, py))]

    # --- 3. MOSTRAR RESULTADOS ---
    
    st.subheader(f"Analizando el instante t = {t_val}")

    # Pestañas para organizar la info
    tab1, tab2, tab3 = st.tabs(["Líneas de Corriente", "Trayectoria", "Línea de Humo"])

    # ==========================================
    # PESTAÑA 1: LÍNEAS DE CORRIENTE (Streamlines)
    # Ecuación: dy/dx = v/u (con t fijo)
    # ==========================================
    with tab1:
        col_math, col_graph = st.columns([1, 2])
        
        with col_math:
            st.markdown("### 📐 Ecuación")
            st.info("Se obtiene resolviendo: $\\frac{dy}{dx} = \\frac{v}{u}$ (t fijo)")
            
            # Intentar resolver simbólicamente
            try:
                # Sustituimos el tiempo por el valor fijo elegido
                u_fixed = u_sym.subs(t, t_val)
                v_fixed = v_sym.subs(t, t_val)
                
                equation = sp.Eq(sp.Derivative(y, x), v_fixed / u_fixed)
                st.latex(sp.latex(equation))
                
                st.write("Solución general (aproximada):")
                sol = sp.dsolve(sp.Function('y')(x).diff(x) - v_fixed/u_fixed, sp.Function('y')(x))
                st.latex(sp.latex(sol))
            except:
                st.warning("La ecuación es demasiado compleja para mostrar la solución analítica automática.")

        with col_graph:
            st.markdown("### 📈 Gráfica del Campo")
            fig, ax = plt.subplots()
            
            # Grid para streamplot
            Y, X = np.mgrid[-5:5:100j, -5:5:100j]
            # Calculamos U y V en toda la malla para el tiempo t_val
            U_num = func_u(t_val, X, Y)
            V_num = func_v(t_val, X, Y)
            
            # Streamplot pinta las líneas de corriente automáticamente
            strm = ax.streamplot(X, Y, U_num, V_num, color=U_num, linewidth=1, cmap='autumn')
            ax.set_title(f"Líneas de Corriente en t={t_val}")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.set_xlim(-5,5); ax.set_ylim(-5,5)
            fig.colorbar(strm.lines)
            st.pyplot(fig)

    # ==========================================
    # PESTAÑA 2: TRAYECTORIA (Pathlines)
    # Ecuación: dx/dt = u, dy/dt = v
    # ==========================================
    with tab2:
        col_math, col_graph = st.columns([1, 2])
        
        with col_math:
            st.markdown("### 📐 Ecuación")
            st.info("Se obtiene integrando: $\\frac{d\\vec{r}}{dt} = \\vec{V}(\\vec{r}, t)$")
            st.write("Sistema de Ecuaciones Diferenciales:")
            st.latex(f"\\frac{{dx}}{{dt}} = {sp.latex(u_sym)}")
            st.latex(f"\\frac{{dy}}{{dt}} = {sp.latex(v_sym)}")
            
        with col_graph:
            st.markdown("### 📈 Gráfica de la Partícula")
            
            # Resolver numéricamente la trayectoria desde t=0 hasta t_val
            t_span = np.linspace(0, t_val, 100)
            
            def model_trayectoria(pos, time_var):
                return get_vel(time_var, pos)
            
            # Integramos
            path = odeint(model_trayectoria, [x0, y0], t_span)
            
            fig2, ax2 = plt.subplots()
            ax2.plot(path[:,0], path[:,1], 'b-', label='Recorrido histórico')
            ax2.plot(path[-1,0], path[-1,1], 'ro', label=f'Posición actual (t={t_val})')
            ax2.plot(x0, y0, 'go', label='Inicio (t=0)')
            
            ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)
            ax2.grid(True)
            ax2.legend()
            ax2.set_title("Trayectoria de una partícula")
            st.pyplot(fig2)

    # ==========================================
    # PESTAÑA 3: LÍNEA DE HUMO (Streaklines)
    # Definición: Lugar geométrico de partículas inyectadas
    # ==========================================
    with tab3:
        col_math, col_graph = st.columns([1, 2])
        
        with col_math:
            st.markdown("### 📐 Explicación")
            st.info("Línea formada por todas las partículas que han pasado por el punto de inyección $(X_0, Y_0)$ en el pasado.")
            st.markdown("""
            Matemáticamente, si la posición de una partícula es $\\vec{r}(t, \\tau)$, donde $\\tau$ es el momento en que se inyectó:
            La línea de humo en el instante $t$ es el conjunto de puntos:
            """)
            st.latex(f"\\vec{{r}}_{{humo}} = \\vec{{r}}(t, \\tau) \\quad \\text{{para }} 0 \\le \\tau \\le t")
            
        with col_graph:
            st.markdown("### 📈 Gráfica de Humo")
            
            # Para dibujar la línea de humo en el instante t_val:
            # Tenemos que resolver muchas trayectorias.
            # Cada punto de la línea de humo es una partícula que salió en un tiempo tau distinto.
            
            taus = np.linspace(0, t_val, 40) # 40 partículas inyectadas en distintos momentos
            humo_x = []
            humo_y = []
            
            for tau in taus:
                # Integramos esta partícula desde SU tiempo de nacimiento (tau) hasta AHORA (t_val)
                if t_val > tau:
                    t_vida = np.linspace(tau, t_val, 20)
                    trayectoria_p = odeint(model_trayectoria, [x0, y0], t_vida)
                    # Nos quedamos solo con la posición final (donde está ahora)
                    pos_final = trayectoria_p[-1]
                    humo_x.append(pos_final[0])
                    humo_y.append(pos_final[1])
                else:
                    humo_x.append(x0)
                    humo_y.append(y0)

            fig3, ax3 = plt.subplots()
            # Dibujamos los puntos conectados
            ax3.plot(humo_x, humo_y, 'o-', color='purple', markersize=4, alpha=0.7)
            ax3.plot(x0, y0, 'r^', label='Inyector', markersize=10)
            
            ax3.set_xlim(-5, 5); ax3.set_ylim(-5, 5)
            ax3.grid(True)
            ax3.set_title(f"Línea de Humo en t={t_val}")
            ax3.legend()
            st.pyplot(fig3)

if __name__ == "__main__":
    main()

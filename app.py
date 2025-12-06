import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulación MMC: Flujo Couette", layout="wide")

st.title("🌊 Simulación de Mecánica de Medios Continuos")
st.markdown("""
**Asignatura:** Mecánica de los Medios Continuos  
**Tema:** Flujo de Couette con Disipación Viscosa (Navier-Stokes + Energía)

Esta aplicación simula cómo se mueve un fluido viscoso entre dos placas planas y cómo aumenta su temperatura debido a la fricción interna (término de disipación del tensor de tensiones).
""")

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("⚙️ Parámetros de Entrada")

# Propiedades Físicas
U_wall = st.sidebar.slider("Velocidad Placa Superior (m/s)", 0.1, 10.0, 5.0)
H = st.sidebar.number_input("Distancia entre placas (m)", 0.001, 0.1, 0.01, format="%.3f")
mu = st.sidebar.number_input("Viscosidad Dinámica (Pa·s)", 0.001, 10.0, 0.8)
rho = st.sidebar.number_input("Densidad (kg/m³)", 100.0, 2000.0, 900.0)
k_thermal = st.sidebar.number_input("Conductividad Térmica (W/m·K)", 0.1, 50.0, 0.15)
cp = st.sidebar.number_input("Calor Específico (J/kg·K)", 100.0, 5000.0, 2000.0)

# Parámetros Numéricos
n_points = st.sidebar.slider("Puntos de malla (Resolución)", 10, 100, 50)
t_max = st.sidebar.slider("Tiempo de simulación (s)", 1.0, 20.0, 5.0)

# Botón de inicio
start_sim = st.sidebar.button("🚀 INICIAR SIMULACIÓN")

# --- LÓGICA DE SIMULACIÓN ---
if start_sim:
    # 1. Discretización (Espacio)
    dy = H / (n_points - 1)
    y_coords = np.linspace(0, H, n_points)
    
    # 2. Cálculo del paso de tiempo estable (Criterio de Courant/Estabilidad)
    # Necesitamos que dt sea pequeño para que la simulación no "explote"
    diffusivity_momentum = mu / rho
    diffusivity_thermal = k_thermal / (rho * cp)
    max_diff = max(diffusivity_momentum, diffusivity_thermal)
    
    # Factor de seguridad 0.4 (debe ser <= 0.5)
    dt = 0.4 * (dy**2) / max_diff
    n_steps = int(t_max / dt)
    
    st.info(f"Calculando... Paso de tiempo dt: {dt:.6f} s | Pasos totales: {n_steps}")

    # 3. Condiciones Iniciales (Todo quieto y a 20ºC)
    u = np.zeros(n_points)          # Velocidad inicial 0
    T = np.ones(n_points) * 20.0    # Temperatura inicial 20ºC
    
    # Espacio para las gráficas
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)

    # 4. Bucle Temporal (Iterativo)
    # Para no saturar, actualizamos la gráfica cada X pasos
    plot_every = max(1, n_steps // 50) 
    
    for n in range(n_steps):
        # Guardamos los valores anteriores
        u_old = u.copy()
        T_old = T.copy()
        
        # --- Ecuación de Cantidad de Movimiento (Navier-Stokes 1D) ---
        # du/dt = nu * d2u/dy2
        # u_new = u_old + dt * nu * (u_i+1 - 2u_i + u_i-1) / dy^2
        laplacian_u = (u_old[2:] - 2*u_old[1:-1] + u_old[:-2]) / dy**2
        u[1:-1] = u_old[1:-1] + dt * diffusivity_momentum * laplacian_u
        
        # Condiciones de Contorno Velocidad
        u[0] = 0.0       # Placa inferior quieta
        u[-1] = U_wall   # Placa superior se mueve
        
        # --- Ecuación de la Energía ---
        # dT/dt = alpha * d2T/dy2 + (mu/rho*cp) * (du/dy)^2
        # El último término es la DISIPACIÓN VISCOSA (Calor por fricción)
        
        # Derivada de velocidad (du/dy) centrada
        du_dy = (u_old[2:] - u_old[:-2]) / (2 * dy)
        viscous_dissipation = (mu / (rho * cp)) * (du_dy ** 2)
        
        laplacian_T = (T_old[2:] - 2*T_old[1:-1] + T_old[:-2]) / dy**2
        T[1:-1] = T_old[1:-1] + dt * diffusivity_thermal * laplacian_T + dt * viscous_dissipation
        
        # Condiciones de Contorno Temperatura (Paredes fijas a 20ºC)
        T[0] = 20.0
        T[-1] = 20.0

        # --- Actualización Gráfica ---
        if n % plot_every == 0 or n == n_steps - 1:
            # Crear figura con dos ejes
            fig = go.Figure()
            
            # Gráfica de Velocidad
            fig.add_trace(go.Scatter(x=y_coords, y=u, mode='lines', name='Velocidad u(y)', line=dict(color='blue', width=3)))
            
            # Gráfica de Temperatura (Eje secundario para visualizar mejor)
            fig.add_trace(go.Scatter(x=y_coords, y=T, mode='lines', name='Temperatura T(y)', line=dict(color='red', width=3), yaxis='y2'))
            
            fig.update_layout(
                title=f"Perfil de Flujo (Tiempo: {n*dt:.3f} s)",
                xaxis_title="Posición en el canal (y) [m]",
                yaxis=dict(title="Velocidad [m/s]", titlefont=dict(color="blue")),
                yaxis2=dict(title="Temperatura [ºC]", titlefont=dict(color="red"), overlaying='y', side='right'),
                template="plotly_white"
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            progress_bar.progress(min(n / n_steps, 1.0))
            
            # Pequeña pausa para ver la animación (solo si son pocos pasos)
            # time.sleep(0.01)

    st.success("✅ Simulación finalizada. Observe cómo el perfil de velocidad se hace lineal y la temperatura sube en el centro.")
    
    # Explicación de resultados
    st.markdown("""
    ### 📝 Interpretación de Resultados
    1. **Velocidad (Azul):** El fluido comienza quieto. Al moverse la placa superior, arrastra al fluido por viscosidad hasta crear un perfil lineal (Flujo de Couette puro).
    2. **Temperatura (Roja):** Debido a la viscosidad, el movimiento genera calor (disipación). Como las paredes están frías (20ºC), el calor se acumula en el centro, creando una parábola de temperatura. **Esto demuestra la conservación de la energía en un medio continuo.**
    """)

else:
    st.info("👈 Ajusta los parámetros en el menú lateral y pulsa 'INICIAR SIMULACIÓN'")

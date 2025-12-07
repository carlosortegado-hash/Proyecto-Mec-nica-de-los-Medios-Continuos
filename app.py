import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de página
st.set_page_config(page_title="Simulador Vórtice + Ascensor", layout="wide")

def main():
    st.title("🌪️ Vórtice Forzado con Aceleración Vertical")
    st.markdown("""
    Este simulador utiliza la expresión analítica de $z_s(r)$ para comprobar si el fluido se derrama o toca el fondo 
    bajo condiciones de rotación y aceleración vertical (Ej. Problema 69 del boletín).
    """)

    # --- 1. PARÁMETROS DE ENTRADA ---
    st.sidebar.header("⚙️ Geometría y Condiciones")
    
    # Geometría del cilindro
    H_cilindro = st.sidebar.number_input("Altura total del cilindro (H) [m]", 0.5, 5.0, 1.5, 0.1)
    R = st.sidebar.number_input("Radio del cilindro (R) [m]", 0.1, 2.0, 0.5, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("💧 Fluido y Movimiento")
    
    # Estado inicial
    h0 = st.sidebar.slider("Nivel inicial de líquido (h0) [m]", 0.1, H_cilindro, H_cilindro*0.6, 0.05)
    
    # Dinámica
    omega = st.sidebar.slider("Velocidad Angular (ω) [rad/s]", 0.0, 20.0, 5.0, 0.1)
    
    # Aceleración externa (Ascensor)
    st.sidebar.markdown("### 🚀 Aceleración Externa")
    st.sidebar.info("Si el ascensor SUBE acelerando, la gravedad aparente aumenta.")
    a_ascensor = st.sidebar.number_input("Aceleración del ascensor (a_z) [m/s²]", -9.0, 20.0, 0.0, 0.5)
    
    # --- 2. CÁLCULOS FÍSICOS (La Expresión de Clase) ---
    
    g_tierra = 9.81
    # Gravedad efectiva: g' = g + a (Principio de equivalencia)
    g_eff = g_tierra + a_ascensor
    
    if g_eff <= 0:
        st.error("⛔ ¡Error Físico! Si la aceleración hacia abajo es mayor que la gravedad, el agua flotaría libremente.")
        return

    # Ecuación de la altura de la superficie libre z_s(r)
    # z_s(r) = (h0 - (omega^2 * R^2)/(4g)) + (omega^2 * r^2)/(2g)
    # El primer término es z_min (altura en el centro)
    
    termino_comun = (omega**2) / (2 * g_eff)
    z_min = h0 - termino_comun * (R**2 / 2) # Esto es h0 - (w^2 R^2)/(4g)
    z_max = z_min + termino_comun * (R**2)  # Esto es la altura en la pared r=R
    
    # --- 3. COMPROBACIONES DE SEGURIDAD ---
    
    col_info, col_graf = st.columns([1, 2])
    
    with col_info:
        st.subheader("📊 Resultados")
        st.write(f"**Gravedad Efectiva ($g'$):** {g_eff:.2f} m/s²")
        
        # Métricas
        st.metric("Altura en el centro ($z_{min}$)", f"{z_min:.3f} m")
        st.metric("Altura en la pared ($z_{max}$)", f"{z_max:.3f} m")
        
        st.markdown("---")
        st.subheader("⚠️ Diagnóstico")
        
        estado_ok = True
        
        # Chequeo 1: ¿Toca el fondo?
        if z_min < 0:
            st.error("❌ **EL FONDO ESTÁ SECO**: El vórtice es tan fuerte que toca el suelo del recipiente.")
            estado_ok = False
        else:
            st.success("✅ Fondo cubierto de agua.")
            
        # Chequeo 2: ¿Se sale por arriba?
        if z_max > H_cilindro:
            st.error(f"❌ **DERRAME**: El agua rebasa la altura del cilindro ({H_cilindro} m).")
            estado_ok = False
        else:
            st.success("✅ El agua no se derrama.")
            
        if estado_ok:
            st.info("El sistema está en equilibrio seguro.")

    # --- 4. VISUALIZACIÓN GRÁFICA ---
    
    with col_graf:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Malla radial
        r = np.linspace(0, R, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)
        
        # Aplicamos la fórmula exacta Z_s(r)
        # Z = z_min + (omega^2 * r^2) / (2g)
        Z = z_min + termino_comun * (r_grid**2)
        
        # Recortamos visualmente si se sale de los límites físicos (para que el dibujo sea realista)
        # Lo que esté por debajo de 0 lo pintamos como 0, lo que esté por encima de H, como H
        Z_visual = np.clip(Z, 0, H_cilindro) 
        
        # Pintar superficie del agua
        surf = ax.plot_surface(X, Y, Z_visual, cmap='winter', alpha=0.7, rstride=2, cstride=2)
        
        # --- DIBUJAR EL CILINDRO (Referencia visual) ---
        # Tapa superior (borde)
        theta_line = np.linspace(0, 2*np.pi, 100)
        x_rim = R * np.cos(theta_line)
        y_rim = R * np.sin(theta_line)
        z_rim = np.full_like(theta_line, H_cilindro)
        ax.plot(x_rim, y_rim, z_rim, color='black', linewidth=3, label='Borde Recipiente')
        
        # Paredes del cilindro (malla de alambre gris)
        z_wall = np.linspace(0, H_cilindro, 10)
        theta_w, z_w = np.meshgrid(theta_line, z_wall)
        x_w = R * np.cos(theta_w)
        y_w = R * np.sin(theta_w)
        ax.plot_surface(x_w, y_w, z_w, color='gray', alpha=0.1)

        # Configuración de ejes
        ax.set_zlim(0, H_cilindro * 1.2)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altura Z [m]')
        ax.set_title(f"Superficie Libre ($a_z$ = {a_ascensor} m/s²)")
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()

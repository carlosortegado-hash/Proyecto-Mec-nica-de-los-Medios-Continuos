import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de página
st.set_page_config(page_title="Simulador Vórtice - Gravedad Variable", layout="wide")

def main():
    st.title("🌪️ Vórtice Forzado con Gravedad Variable")
    st.markdown("""
    Este simulador calcula la superficie libre de un fluido en rotación $z_s(r)$.
    Puedes modificar la **Gravedad (g)** para simular condiciones en otros planetas 
    o sistemas acelerados (gravedad efectiva).
    """)

    # --- 1. PARÁMETROS DE ENTRADA ---
    st.sidebar.header("Geometría del Recipiente")
    
    # Geometría del cilindro
    H_cilindro = st.sidebar.number_input("Altura total del cilindro (H) [m]", 0.5, 5.0, 1.5, 0.1)
    R = st.sidebar.number_input("Radio del cilindro (R) [m]", 0.1, 2.0, 0.5, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("💧 Condiciones Físicas")
    
    # Estado inicial del fluido
    h0 = st.sidebar.slider("Nivel inicial de líquido (h0) [m]", 0.1, H_cilindro, H_cilindro*0.6, 0.05)
    
    # Variables Dinámicas
    omega = st.sidebar.slider("Velocidad Angular (ω) [rad/s]", 0.0, 20.0, 5.0, 0.1)
    
    # SELECCIÓN DIRECTA DE GRAVEDAD
    st.sidebar.markdown("### 🪐 Gravedad")
    g_input = st.sidebar.number_input("Aceleración de la gravedad (g) [m/s²]", 0.1, 50.0, 9.81, 0.1)
    st.sidebar.caption("Ejemplos: Tierra=9.81, Luna=1.62, Marte=3.71")

    # --- 2. CÁLCULOS FÍSICOS ---
    
    # Usamos directamente el input de gravedad
    g = g_input
    
    # Ecuación de la altura de la superficie libre z_s(r) derivada en clase:
    # z_s(r) = z_min + (omega^2 * r^2) / (2g)
    # Por conservación de volumen, z_min se relaciona con h0:
    
    termino_comun = (omega**2) / (2 * g)
    z_min = h0 - termino_comun * (R**2 / 2) 
    z_max = z_min + termino_comun * (R**2)  
    
    # --- 3. DIAGNÓSTICO Y ALERTAS ---
    
    col_info, col_graf = st.columns([1, 2])
    
    with col_info:
        st.subheader("📊 Resultados Analíticos")
        
        # Métricas principales
        colA, colB = st.columns(2)
        with colA:
            st.metric("Altura Centro ($z_{min}$)", f"{z_min:.3f} m")
        with colB:
            st.metric("Altura Pared ($z_{max}$)", f"{z_max:.3f} m")
        
        st.markdown("---")
        st.subheader("⚠️ Estado del Sistema")
        
        estado_ok = True
        
        # Alerta 1: Fondo seco
        if z_min < 0:
            st.error("❌ **FONDO SECO**: El vórtice toca el suelo. La ecuación deja de ser válida en el centro seco.")
            estado_ok = False
        else:
            st.success("✅ Fondo cubierto.")
            
        # Alerta 2: Derrame
        if z_max > H_cilindro:
            st.error(f"❌ **DERRAME**: El líquido rebasa la altura {H_cilindro} m.")
            estado_ok = False
        else:
            st.success("✅ Sin derrame.")
            
        if estado_ok:
            st.info("El sistema está estable y contenido.")

    # --- 4. VISUALIZACIÓN 3D ---
    
    with col_graf:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generar malla cilíndrica
        r = np.linspace(0, R, 40)
        theta = np.linspace(0, 2*np.pi, 60)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)
        
        # Calcular alturas Z
        Z = z_min + termino_comun * (r_grid**2)
        
        # Clip visual (para que el dibujo no se salga de la caja gráfica de forma fea)
        Z_visual = np.clip(Z, 0, H_cilindro) 
        
        # 1. Dibujar Fluido
        surf = ax.plot_surface(X, Y, Z_visual, cmap='winter', alpha=0.7, rstride=2, cstride=2, antialiased=True)
        
        # 2. Dibujar Estructura del Cilindro (Visualización)
        # Borde superior
        theta_line = np.linspace(0, 2*np.pi, 100)
        x_rim = R * np.cos(theta_line)
        y_rim = R * np.sin(theta_line)
        z_rim = np.full_like(theta_line, H_cilindro)
        ax.plot(x_rim, y_rim, z_rim, color='black', linewidth=3, label='Borde')
        
        # Paredes transparentes
        z_wall = np.linspace(0, H_cilindro, 2) # Solo base y tapa para aligerar
        theta_w, z_w = np.meshgrid(theta_line, z_wall)
        x_w = R * np.cos(theta_w)
        y_w = R * np.sin(theta_w)
        ax.plot_surface(x_w, y_w, z_w, color='gray', alpha=0.15)

        # Ajustes de la cámara y ejes
        ax.set_zlim(0, H_cilindro * 1.1)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altura Z [m]')
        ax.set_title(f"Superficie Libre (g = {g} m/s²)")
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de la página
st.set_page_config(page_title="Superficie Libre Exacta", layout="wide")

def main():
    st.title("🌪️ Superficie Libre: Ecuación Exacta")
    st.markdown("Visualización basada en la conservación de volumen en un recipiente cilíndrico.")

    # --- 1. BARRA LATERAL (DATOS) ---
    st.sidebar.header("1. Geometría del Cilindro")
    H_cilindro = st.sidebar.number_input("Altura total del recipiente (H) [m]", 0.5, 10.0, 1.5, 0.1)
    R = st.sidebar.number_input("Radio del recipiente (R) [m]", 0.1, 5.0, 0.5, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Condiciones Físicas")
    h0 = st.sidebar.number_input("Altura inicial del líquido (h0) [m]", 0.1, H_cilindro, 0.8, 0.05)
    omega = st.sidebar.slider("Velocidad angular (ω) [rad/s]", 0.0, 25.0, 5.0, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. Entorno")
    g = st.sidebar.number_input("Gravedad (g) [m/s²]", 0.1, 30.0, 9.81, 0.01)

    # --- 2. MOSTRAR LA FÓRMULA UTILIZADA ---
    st.markdown("### 📐 Ecuación aplicada:")
    st.latex(r"z(r) = h_0 - \frac{\omega^2 R^2}{4g} + \frac{\omega^2 r^2}{2g}")

    # --- 3. CÁLCULO DE PUNTOS CLAVE (Usando la fórmula exacta) ---
    # Calculamos los puntos extremos para el diagnóstico numérico
    
    # Altura en el centro (r = 0)
    z_min = h0 - (omega**2 * R**2) / (4 * g) + 0 
    
    # Altura en la pared (r = R)
    z_max = h0 - (omega**2 * R**2) / (4 * g) + (omega**2 * R**2) / (2 * g)
    
    # --- 4. DIAGNÓSTICO DE SEGURIDAD ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Valores Calculados")
        st.write(f"**Altura inicial ($h_0$):** {h0} m")
        st.metric("Altura mínima (Centro)", f"{z_min:.3f} m")
        st.metric("Altura máxima (Pared)", f"{z_max:.3f} m")

        st.markdown("---")
        st.subheader("⚠️ Comprobación")
        
        estado = "OK"
        if z_min < 0:
            st.error("❌ **FONDO SECO**: El líquido toca el fondo del recipiente ($z < 0$).")
            estado = "ERROR"
        
        if z_max > H_cilindro:
            st.error(f"❌ **DERRAME**: El líquido rebosa por arriba (Altura > {H_cilindro} m).")
            estado = "ERROR"
            
        if estado == "OK":
            st.success("✅ Sistema estable: El líquido se mantiene dentro del recipiente.")

    # --- 5. GRÁFICA 3D ---
    with col2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Malla de coordenadas
        r = np.linspace(0, R, 50)
        theta = np.linspace(0, 2*np.pi, 60)
        r_grid, theta_grid = np.meshgrid(r, theta)

        # Transformación a cartesianas
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)

        # -------------------------------------------------------
        # APLICACIÓN DE LA FÓRMULA EXACTA EN LA MALLA
        # -------------------------------------------------------
        Z = h0 - (omega**2 * R**2) / (4 * g) + (omega**2 * r_grid**2) / (2 * g)
        
        # Recorte visual (Clip) para que el gráfico sea realista
        # Si z < 0 se ve el fondo (0), si z > H se corta en H
        Z_visual = np.clip(Z, 0, H_cilindro)

        # Dibujar superficie del agua
        surf = ax.plot_surface(X, Y, Z_visual, cmap='Blues', alpha=0.8, antialiased=True)

        # Dibujar referencia del cilindro
        # 1. Borde superior
        theta_line = np.linspace(0, 2*np.pi, 100)
        x_rim = R * np.cos(theta_line)
        y_rim = R * np.sin(theta_line)
        z_rim = np.full_like(theta_line, H_cilindro)
        ax.plot(x_rim, y_rim, z_rim, color='black', linewidth=2, label='Borde Recipiente')

        # 2. Línea de nivel inicial (referencia visual)
        z_h0 = np.full_like(theta_line, h0)
        ax.plot(x_rim, y_rim, z_h0, color='red', linestyle='--', linewidth=1, label=f'Nivel inicial h0={h0}')

        # Ajustes de ejes
        ax.set_zlim(0, H_cilindro * 1.1)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altura Z [m]')
        ax.set_title(f"Perfil del Fluido (ω = {omega} rad/s)")
        
        # Leyenda manual para entender las líneas
        ax.text2D(0.05, 0.95, "Azul: Superficie libre\nRojo: Nivel inicial", transform=ax.transAxes)

        st.pyplot(fig)

if __name__ == "__main__":
    main()

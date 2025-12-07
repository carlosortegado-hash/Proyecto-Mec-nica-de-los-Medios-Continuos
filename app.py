import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de la página
st.set_page_config(page_title="Simulador de Fluidos", layout="wide")

def main():
    st.title("🌪️ Superficie Libre en Rotación")
    st.markdown("Simulación basada en la conservación de volumen para un recipiente cilíndrico.")

    # --- 1. PARÁMETROS (Igual que antes) ---
    st.sidebar.header("1. Geometría")
    H_cilindro = st.sidebar.number_input("Altura del cilindro (H) [m]", 0.1, 10.0, 1.5, 0.1)
    R = st.sidebar.number_input("Radio del cilindro (R) [m]", 0.1, 5.0, 0.5, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Condiciones")
    h0 = st.sidebar.number_input("Altura inicial (h0) [m]", 0.1, H_cilindro, 0.8, 0.05)
    omega = st.sidebar.slider("Velocidad angular (ω) [rad/s]", 0.0, 25.0, 5.0, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. Gravedad")
    g = st.sidebar.number_input("Gravedad (g) [m/s²]", 0.1, 30.0, 9.81, 0.1)

    # --- 2. FÓRMULA EXACTA ---
    st.markdown("### 📐 Expresión utilizada:")
    st.latex(r"z(r) = h_0 - \frac{\omega^2 R^2}{4g} + \frac{\omega^2 r^2}{2g}")

    # --- 3. CÁLCULOS (Usando TU fórmula estrictamente) ---
    
    # Altura en el centro (r=0)
    # El término (w^2 * r^2) / 2g se hace cero.
    z_min = h0 - (omega**2 * R**2) / (4 * g)
    
    # Altura en la pared (r=R)
    z_max = h0 - (omega**2 * R**2) / (4 * g) + (omega**2 * R**2) / (2 * g)
    
    # --- 4. RESULTADOS Y ALERTAS ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Datos Calculados")
        st.write(f"**Altura inicial ($h_0$):** {h0} m")
        st.metric("Altura mínima (Centro)", f"{z_min:.3f} m")
        st.metric("Altura máxima (Pared)", f"{z_max:.3f} m")

        st.markdown("---")
        st.subheader("⚠️ Comprobación")
        
        estado_ok = True
        
        # Comprobación 1: Fondo seco
        if z_min < 0:
            st.error("❌ **FONDO SECO**: El fluido toca el fondo (z < 0).")
            estado_ok = False
            
        # Comprobación 2: Derrame
        if z_max > H_cilindro:
            st.error(f"❌ **DERRAME**: El fluido rebosa la altura {H_cilindro} m.")
            estado_ok = False
            
        if estado_ok:
            st.success("✅ Sistema en equilibrio dentro del recipiente.")

    # --- 5. GRÁFICA 3D ---
    with col2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Malla
        r = np.linspace(0, R, 50)
        theta = np.linspace(0, 2*np.pi, 60)
        r_grid, theta_grid = np.meshgrid(r, theta)

        # Coordenadas X, Y
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)

        # Coordenada Z (Aplicando TU fórmula a la malla)
        # z = h0 - term1 + term2
        term1 = (omega**2 * R**2) / (4 * g)
        term2 = (omega**2 * r_grid**2) / (2 * g)
        Z = h0 - term1 + term2
        
        # Visualmente cortamos el agua si se sale de los límites físicos
        Z_visual = np.clip(Z, 0, H_cilindro)

        # Dibujar Agua
        ax.plot_surface(X, Y, Z_visual, cmap='Blues', alpha=0.8, antialiased=True)

        # Dibujar Referencias (Cilindro)
        # Borde superior
        theta_line = np.linspace(0, 2*np.pi, 100)
        x_rim = R * np.cos(theta_line)
        y_rim = R * np.sin(theta_line)
        z_rim = np.full_like(theta_line, H_cilindro)
        ax.plot(x_rim, y_rim, z_rim, color='black', linewidth=2, label='Borde')

        # Nivel inicial (Línea roja)
        z_h0 = np.full_like(theta_line, h0)
        ax.plot(x_rim, y_rim, z_h0, color='red', linestyle='--', label='Nivel Inicial')

        ax.set_zlim(0, H_cilindro * 1.1)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altura Z [m]')
        ax.set_title(f"Perfil del Fluido (g={g})")
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()

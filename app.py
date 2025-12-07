import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de la página
st.set_page_config(page_title="Simulador de Fluidos", layout="wide")

def main():
    st.title("Superficie Libre en Rotación")
    st.markdown("Simulación que calcula y representa el perfil parabólico de la superficie libre de un fluido contenido en un recipiente cilíndrico que rota a velocidad angular constante.")

    # --- 1. PARÁMETROS ---
    st.sidebar.header("1. Geometría del cilindro")
    H_cilindro = st.sidebar.number_input("Altura del cilindro (H) [m]", 0.1, 10.0, 1.5, 0.1)
    R = st.sidebar.number_input("Radio del cilindro (R) [m]", 0.1, 5.0, 0.5, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Condiciones físicas")
    h0 = st.sidebar.number_input("Altura inicial del fluido (h) [m]", 0.1, H_cilindro, 0.8, 0.05)
    omega = st.sidebar.slider("Velocidad angular (ω) [rad/s]", 0.0, 25.0, 5.0, 0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("3. Gravedad efectiva")
    g = st.sidebar.number_input("Gravedad efectiva (g) [m/s²]", 0.1, 30.0, 9.81, 0.1)

    # --- 2. FÓRMULA ---
    st.markdown("Expresión utilizada:")
    st.latex(r"z(r) = h - \frac{\omega^2 R^2}{4g} + \frac{\omega^2 r^2}{2g}")

    # --- 3. CÁLCULOS ---
    
    # Altura en el centro (r=0)
    z_min = h0 - (omega**2 * R**2) / (4 * g)
    
    # Altura en la pared (r=R)
    z_max = h0 - (omega**2 * R**2) / (4 * g) + (omega**2 * R**2) / (2 * g)
    
    # --- 4. RESULTADOS Y ALERTAS ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Datos Calculados")
        st.metric("Altura mínima (Centro)", f"{z_min:.3f} m")
        st.metric("Altura máxima (Pared)", f"{z_max:.3f} m")

        st.markdown("---")
        st.subheader("Comprobación")
        
        estado_ok = True
        if z_min < 0:
            st.error("**FONDO SECO**: El fluido toca el fondo (z < 0)")
            estado_ok = False
        if z_max > H_cilindro:
            st.error(f"**DERRAME**: El fluido rebosa del recipiente")
            estado_ok = False   
        if estado_ok:
            st.success("Sistema en equilibrio dentro del recipiente.")

    # --- 5. GRÁFICA 3D ---
    with col2:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection='3d')

        # --- A. EL FLUIDO ---
        # Malla radial para el agua
        r = np.linspace(0, R, 50)
        theta = np.linspace(0, 2*np.pi, 60)
        r_grid, theta_grid = np.meshgrid(r, theta)

        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)

        # Fórmula exacta
        term1 = (omega**2 * R**2) / (4 * g)
        term2 = (omega**2 * r_grid**2) / (2 * g)
        Z = h0 - term1 + term2
        
        # Clip visual para que no atraviese el suelo ni salga del techo en el dibujo
        Z_visual = np.clip(Z, 0, H_cilindro)

        # Pintar Agua
        ax.plot_surface(X, Y, Z_visual, cmap='Blues', alpha=0.8, antialiased=True)

        # --- B. EL RECIPIENTE (Paredes y Base) ---
        
        # 1. Paredes Laterales (Cilindro hueco)
        z_walls = np.linspace(0, H_cilindro, 20)
        theta_walls = np.linspace(0, 2*np.pi, 50)
        theta_w_grid, z_w_grid = np.meshgrid(theta_walls, z_walls)
        
        x_w = R * np.cos(theta_w_grid)
        y_w = R * np.sin(theta_w_grid)
        
        # Pintamos las paredes de color gris transparente (alpha=0.15)
        ax.plot_surface(x_w, y_w, z_w_grid, color='gray', alpha=0.15)
        
        # 2. Base del cilindro (Disco en z=0)
        # Reutilizamos la malla X, Y que usamos para el agua, pero con Z=0
        Z_bottom = np.zeros_like(X)
        ax.plot_surface(X, Y, Z_bottom, color='black', alpha=0.3)

        # 3. Borde superior (Aro negro)
        theta_line = np.linspace(0, 2*np.pi, 100)
        x_rim = R * np.cos(theta_line)
        y_rim = R * np.sin(theta_line)
        z_rim = np.full_like(theta_line, H_cilindro)
        ax.plot(x_rim, y_rim, z_rim, color='black', linewidth=3, label='Borde Superior')

        # --- AJUSTES ---
        ax.set_zlim(0, H_cilindro * 1.1)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Altura Z [m]')
        ax.set_title(f"Visualización 3D (g={g})")
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()

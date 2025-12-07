import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración
st.set_page_config(page_title="Simulador Vórtice Forzado", layout="wide")

def main():
    st.title("🌪️ Simulador de Líquido en Rotación")
    st.markdown("""
    Esta simulación visualiza los **Problemas 67-69** de tu boletín.
    
    Al girar un recipiente cilíndrico, la superficie libre adopta la forma de un **paraboloide de revolución** debido al equilibrio entre la gravedad y la fuerza centrífuga.
    """)

    # --- CONTROLES LATERALES ---
    st.sidebar.header("Parámetros del Experimento")
    
    # Sliders para jugar con las variables
    omega = st.sidebar.slider("Velocidad Angular (rad/s)", 0.0, 15.0, 5.0, 0.1)
    R = st.sidebar.slider("Radio del Recipiente (m)", 0.1, 1.0, 0.5, 0.1)
    h0 = st.sidebar.slider("Nivel inicial de agua (m)", 0.1, 2.0, 1.0, 0.1)
    
    # Constante g
    g = 9.81

    # --- CÁLCULOS FÍSICOS (Sencillos) ---
    # 1. Calculamos la altura en el centro (z_min) usando conservación de volumen
    # El volumen inicial es pi*R^2*h0.
    # El volumen del paraboloide se ajusta para que el promedio sea h0.
    # Fórmula: z(r) = z_min + (w^2 * r^2) / (2g)
    # Tras integrar, sabemos que la diferencia de altura entre pared y centro es: Delta_z = (w^2 R^2) / (2g)
    # Y el nivel desciende en el centro la mitad de esa diferencia:
    
    delta_z = (omega**2 * R**2) / (2*g)
    z_min = h0 - delta_z / 2
    z_max = h0 + delta_z / 2
    
    # --- VISUALIZACIÓN ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Datos Calculados")
        st.metric("Altura en la pared (máx)", f"{z_max:.2f} m")
        st.metric("Altura en el centro (mín)", f"{z_min:.2f} m")
        st.metric("Diferencia de altura", f"{delta_z:.2f} m")
        
        if z_min < 0:
            st.error("⚠️ ¡Cuidado! El fondo del recipiente quedaría seco (el vórtice toca el suelo).")
        else:
            st.success("El líquido cubre todo el fondo.")

    with col2:
        st.subheader("Vista 3D del Fluido")
        
        # Crear malla para el gráfico 3D
        r = np.linspace(0, R, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        r_grid, theta_grid = np.meshgrid(r, theta)
        
        # Coordenadas X, Y
        X = r_grid * np.cos(theta_grid)
        Y = r_grid * np.sin(theta_grid)
        
        # Coordenada Z (La ecuación del paraboloide)
        Z = z_min + (omega**2 * r_grid**2) / (2*g)
        
        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Superficie del agua
        surf = ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.8, edgecolor='none')
        
        # Dibujar el recipiente (paredes transparentes) como referencia visual
        z_cilindro = np.linspace(0, max(z_max, h0)*1.2, 10)
        theta_cil = np.linspace(0, 2*np.pi, 30)
        theta_grid_cil, z_grid_cil = np.meshgrid(theta_cil, z_cilindro)
        x_cil = R * np.cos(theta_grid_cil)
        y_cil = R * np.sin(theta_grid_cil)
        ax.plot_surface(x_cil, y_cil, z_grid_cil, color='gray', alpha=0.1)

        # Ajustes del gráfico
        ax.set_zlim(0, max(z_max, h0)*1.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Altura Z (m)')
        ax.set_title(f"Superficie Libre ($\omega$ = {omega} rad/s)")
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()

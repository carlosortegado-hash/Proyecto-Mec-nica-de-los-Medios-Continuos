import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Teorema del Transporte de Reynolds", layout="wide")

# Título y Ecuación
st.title("🌊 Teorema del Transporte de Reynolds (TTR)")
st.markdown(r"""
Esta simulación visualiza la relación entre la variación en un **Volumen de Control (VC)** fijo y los flujos a través de sus fronteras.
La ecuación general para una propiedad extensiva $B$ (y su intensiva $\beta$) es:

$$
\frac{d B_{sis}}{dt} = \frac{\partial}{\partial t} \int_{VC} \rho \beta \, dV + \int_{SC} \rho \beta (\vec{v} \cdot \vec{n}) \, dA
$$

En nuestro caso 1D simplificado (densidad constante, velocidad constante), analizamos el cambio de una propiedad $C(x,t)$ dentro del VC:
$$
\text{Acumulación en VC} = \text{Entrada} - \text{Salida}
$$
""")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("Parámetros de la Simulación")

velocity = st.sidebar.slider("Velocidad del Flujo (v)", 0.1, 2.0, 1.0, 0.1)
time_step = st.sidebar.slider("Tiempo (t)", 0.0, 10.0, 0.0, 0.1)
width = st.sidebar.slider("Ancho de la 'Nube' (Dispersión)", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Leyenda:**")
st.sidebar.markdown("🟦 **Azul:** Volumen de Control (Fijo)")
st.sidebar.markdown("🔴 **Rojo:** Propiedad Transportada (Móvil)")

# --- CÁLCULOS FÍSICOS ---

# Dominio espacial
x = np.linspace(0, 20, 500)
dx = x[1] - x[0]

# Definición del Volumen de Control (VC) fijo entre x=8 y x=12
vc_start = 8.0
vc_end = 12.0

# Función de la propiedad (una campana de Gauss moviéndose)
# C(x,t) representa la concentración o densidad de la propiedad
center_start = 4.0 # Empieza a la izquierda
center_current = center_start + velocity * time_step
concentration = np.exp(-0.5 * ((x - center_current) / width)**2)

# Cálculos de Integrales y Flujos
# 1. Cantidad total dentro del VC (Integral)
indices_vc = np.where((x >= vc_start) & (x <= vc_end))
amount_in_vc = np.trapz(concentration[indices_vc], x[indices_vc])

# 2. Flujos en las fronteras (Flux = v * Concentración)
# Interpolamos para obtener el valor exacto en los bordes del VC
c_inlet = np.interp(vc_start, x, concentration)
c_outlet = np.interp(vc_end, x, concentration)

flux_in = velocity * c_inlet   # Ganancia (entra por la izquierda)
flux_out = velocity * c_outlet # Pérdida (sale por la derecha)
net_change = flux_in - flux_out # Tasa de cambio instantánea

# --- VISUALIZACIÓN ---

# Crear columnas para métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tiempo", f"{time_step:.1f} s")
col2.metric("Propiedad en VC (Integral)", f"{amount_in_vc:.2f}")
col3.metric("Flujo de Entrada (+)", f"{flux_in:.3f}", delta_color="normal")
col4.metric("Flujo de Salida (-)", f"{flux_out:.3f}", delta_color="inverse")

# Gráfico Principal
fig = go.Figure()

# 1. Dibujar la "Nube" de propiedad (Sistema móvil)
fig.add_trace(go.Scatter(
    x=x, y=concentration,
    mode='lines',
    name='Propiedad Transportada (Sistema)',
    line=dict(color='red', width=3),
    fill='tozeroy',
    fillcolor='rgba(255, 0, 0, 0.1)'
))

# 2. Dibujar el Volumen de Control (Caja fija)
fig.add_shape(
    type="rect",
    x0=vc_start, y0=0, x1=vc_end, y1=1.1,
    line=dict(color="RoyalBlue", width=2, dash="dash"),
    fillcolor="rgba(65, 105, 225, 0.1)",
)

# Añadir anotaciones para el VC
fig.add_annotation(x=(vc_start+vc_end)/2, y=1.15, text="Volumen de Control (Fijo)", showarrow=False)

# Añadir flechas de flujo dinámicas
if flux_in > 0.01:
    fig.add_annotation(
        x=vc_start, y=c_inlet,
        text=f"Entrada\n{flux_in:.2f}",
        showarrow=True, arrowhead=2, ax=-40, ay=-40, arrowcolor="green"
    )

if flux_out > 0.01:
    fig.add_annotation(
        x=vc_end, y=c_outlet,
        text=f"Salida\n{flux_out:.2f}",
        showarrow=True, arrowhead=2, ax=40, ay=-40, arrowcolor="red"
    )

fig.update_layout(
    title="Simulación 1D: Paso de una propiedad a través de un VC",
    xaxis_title="Posición (x)",
    yaxis_title="Intensidad de la Propiedad (C)",
    yaxis=dict(range=[0, 1.3]),
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Explicación contextual
st.info(f"""
**Interpretación en t = {time_step}:**
* El **Término Convectivo** (Integral de superficie) es la diferencia neta entre lo que entra y lo que sale: **{flux_in:.2f} - {flux_out:.2f} = {net_change:.2f}**.
* Si este valor es **positivo**, la propiedad se está acumulando dentro de la caja azul.
* Si es **negativo**, la caja se está vaciando.
""")

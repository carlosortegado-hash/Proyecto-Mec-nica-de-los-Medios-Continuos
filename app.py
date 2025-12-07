import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import math

st.set_page_config(layout="wide", page_title="Simulador: Vaciado de depósito")

st.title("Simulador visual de vaciado de depósitos — Conservación de la masa")

# -----------------------
# Panel lateral: entradas
# -----------------------
with st.sidebar:
    st.header("Parámetros de la simulación")

    geom = st.selectbox("Geometría del depósito", ["Cilíndrico", "Troncocónico (frustum)"])

    H0 = st.number_input("Altura inicial del líquido H (m)", min_value=0.01, value=1.0, step=0.1, format="%.3f")
    g = st.number_input("Gravedad g (m/s²)", min_value=0.0, value=9.81, step=0.01, format="%.3f")

    if geom == "Cilíndrico":
        R = st.number_input("Radio del depósito R (m)", min_value=1e-4, value=0.3, step=0.05, format="%.4f")
        def area_top(h): return math.pi * R**2
        A_descr = f"A(h) = π·{R:.3f}² = {math.pi*R**2:.4f} m²"
    else:
        # frustum dimensions: assume small radius at bottom r0, top radius r1 (when full)
        r_bottom = st.number_input("Radio inferior del depósito r_bottom (m)", min_value=1e-4, value=0.1, step=0.05, format="%.4f")
        r_top = st.number_input("Radio superior del depósito r_top (m)", min_value=1e-4, value=0.3, step=0.05, format="%.4f")
        H_geom = st.number_input("Altura geométrica del depósito H_geom (m) (≥ H)", min_value=0.01, value=max(1.0, H0), format="%.3f")
        # linear interpolation of radius with height measured from bottom (0) to H_geom
        def radius_at(h):
            # clamp h in [0,H_geom]
            hh = min(max(h, 0.0), H_geom)
            return r_bottom + (r_top - r_bottom) * (hh / H_geom)
        def area_top(h): 
            return math.pi * radius_at(h)**2
        A_descr = f"Área A(h)=π·r(h)², r(h) lineal entre {r_bottom:.3f}→{r_top:.3f} m over H_geom={H_geom:.3f} m"

    A2 = st.number_input("Área del orificio A₂ (m²)", min_value=1e-8, value=1e-4, format="%.6f")
    Cd = st.number_input("Coeficiente de descarga C_d (adim.)", min_value=0.0, value=0.62, step=0.01, format="%.3f")
    Tsim = st.number_input("Duración de la simulación a mostrar (s) — animación real", min_value=1.0, value=10.0, step=1.0)
    fps = st.slider("FPS aproximado de la animación", 5, 30, 12)
    density = st.number_input("Densidad del líquido ρ (kg/m³)", min_value=1.0, value=1000.0, step=1.0)

    start = st.button("Iniciar simulación")

st.markdown("---")

# -----------------------
# Funciones físicas
# -----------------------
def dh_dt(t, h):
    """dh/dt = - (C_d A2 / A(h)) * sqrt(2 g h)"""
    Ah = area_top(h) if geom == "Troncocónico (frustum)" or geom=="Troncocónico (frustum)" else area_top(h)
    return - (Cd * A2 / Ah) * math.sqrt(2 * g * max(h, 0.0))

# analytic solution for cylindrical case:
def analytic_h_cyl(H, A2, A1, Cd, g, t):
    # h(t) = [ sqrt(H) - (Cd*A2/A1)*sqrt(g/2) * t ]^2, until zero
    coef = (Cd * A2 / A1) * math.sqrt(g/2.0)
    val = max(0.0, (math.sqrt(H) - coef * t))
    return val * val

# -----------------------
# Pre-simulate ODE (physical time) up to drain or generous Tmax
# -----------------------
if start:
    # integrate dh/dt from t=0 to a reasonable t_max until empty
    t_max_guess = max(5.0, Tsim*5)  # allow some margin
    # integrate until h reaches zero with event
    def event_h_zero(t, y):
        return y[0]
    event_h_zero.terminal = True
    event_h_zero.direction = -1

    # wrapper for solve_ivp using numeric area function
    def ode_wrapper(t, y):
        h = float(y[0])
        # choose area depending on geometry
        if geom == "Cilíndrico":
            Ah = area_top(h)
        else:
            Ah = area_top(h)  # uses radius_at internally
        return [ - (Cd * A2 / Ah) * math.sqrt(2 * g * max(h, 0.0)) ]

    sol = solve_ivp(ode_wrapper, [0, t_max_guess], [H0], events=event_h_zero, max_step=0.1, rtol=1e-6)
    t_phys = sol.t
    h_phys = sol.y[0]
    # if not reached zero, extend t array artificially with last value
    if t_phys[-1] < t_max_guess and (len(sol.t_events) == 0 or len(sol.t_events[0]) == 0):
        # no event triggered (unlikely). proceed with solution as is.
        pass

    # compute physical flow (Q) and potential energy over time
    Q_phys = []
    Epot = []
    for hh in h_phys:
        Ah = area_top(hh)
        q = Cd * A2 * math.sqrt(2 * g * max(hh, 0.0))  # volumetric flow (m^3/s)
        Q_phys.append(q)
        # potential energy of fluid column relative to bottom: ρ*g*V*centroid_height
        # For cylinder centroid at h/2 -> V = A1 * h
        if geom == "Cilíndrico":
            V = area_top(hh) * hh
            Epot.append(density * g * V * (hh/2.0))
        else:
            # approximate centroid height for frustum: for a frustum the centroid formula is more involved;
            # here use numerical: compute small slices to approximate V and centroid
            Nslice = 40
            zs = np.linspace(0, hh, Nslice+1)
            vols = []
            moments = []
            for i in range(Nslice):
                zmid = 0.5*(zs[i]+zs[i+1])
                rmid = radius_at(zmid)
                dV = math.pi*rmid*rmid*(zs[i+1]-zs[i])
                vols.append(dV)
                moments.append(dV * zmid)
            V = sum(vols)
            centroid = sum(moments)/V if V>0 else 0.0
            Epot.append(density * g * V * centroid)

    # create interpolation functions for animation sampling
    from scipy.interpolate import interp1d
    t_sample = np.linspace(0, max(t_phys), max( int(max(t_phys)/(1e-3)), 200) )
    h_interp = interp1d(t_phys, h_phys, bounds_error=False, fill_value=(h_phys[0], 0.0))
    q_interp = interp1d(t_phys, Q_phys, bounds_error=False, fill_value=(Q_phys[0], 0.0))
    e_interp = interp1d(t_phys, Epot, bounds_error=False, fill_value=(Epot[0], 0.0))

    # -----------------------
    # Layout: left: animation, right: plots (3)
    # -----------------------
    col1, col2 = st.columns([1,1.2])

    anim_placeholder = col1.empty()
    info_placeholder = col1.empty()
    plots_place = col2.empty()

    # show some textual results
    # time to empty (physical)
    t_empty = t_phys[-1]
    info_placeholder.markdown(f"**Tiempo de vaciado (físico)** ≈ **{t_empty:.3f} s** (tiempo hasta h=0 según ODE numérica).")
    if geom == "Cilíndrico":
        A1 = area_top(0.0)
        # formula analytic T_empty
        T_analytic = math.sqrt(H0) / ((Cd * A2 / A1) * math.sqrt(g/2.0))
        info_placeholder.markdown(f"**Solución analítica (cilíndro):**  \n" 
                                  r"$h(t) = \bigg(\sqrt{H} - \frac{C_d A_2}{A_1}\sqrt{\frac{g}{2}}\;t\bigg)^2$" + "\n\n"
                                  f"Tiempo teórico de vaciado T = {T_analytic:.3f} s")
    else:
        info_placeholder.markdown("Geometría troncocónica: no hay expresión simple cerrada para h(t); se integra numéricamente.")

    # Animation loop: map physical time [0, t_phys[-1]] onto display time Tsim
    frames = int(Tsim * fps)
    phys_times_for_frames = np.linspace(0, t_phys[-1], frames)

    # Prepare fixed figure sizes
    for ti in phys_times_for_frames:
        hh = float(h_interp(ti))
        qq = float(q_interp(ti))
        ee = float(e_interp(ti))

        # --- animation figure: draw tank and fluid level ---
        fig_anim, ax_anim = plt.subplots(figsize=(4,6))
        ax_anim.set_xlim(-0.6, 0.6)
        ax_anim.set_ylim(-0.2, 1.05 * max(H0, 1.0))
        ax_anim.axis('off')

        # Draw tank outline (simple rectangle), represent radius scale horizontally
        # choose visual width proportional to top radius
        if geom == "Cilíndrico":
            disp_width = 0.4
            left = -disp_width
            right = disp_width
            bottom = 0.0
            top = H0
            rect = plt.Rectangle((left, bottom), 2*disp_width, top, fill=False, linewidth=2)
            ax_anim.add_patch(rect)
            # fill water
            water = plt.Rectangle((left, bottom), 2*disp_width, hh, color='royalblue', alpha=0.6)
            ax_anim.add_patch(water)
        else:
            # draw frustum as trapezoid
            # compute visual radii at bottom and top proportional to real radii
            rb = radius_at(0)
            rt = radius_at(H_geom)
            # scale to display width
            scale = 0.8 / max(rb, rt)
            left_bot = -rb*scale
            right_bot = rb*scale
            left_top = -rt*scale
            right_top = rt*scale
            # polygon outline
            ax_anim.plot([left_bot, left_top], [0, H_geom], color='k', linewidth=2)
            ax_anim.plot([right_bot, right_top], [0, H_geom], color='k', linewidth=2)
            # fill water as polygon up to hh
            # compute left/right at hh
            r_hh = radius_at(hh)
            left_h = -r_hh*scale
            right_h = r_hh*scale
            xs = [left_bot, left_h, left_h, left_bot]
            ys = [0, 0, hh, hh]
            ax_anim.fill_between([left_h, right_h], 0, hh, color='royalblue', alpha=0.6)
            ax_anim.set_ylim(-0.2, 1.05 * max(H_geom, H0))

        # draw orifice and jet
        # orifice center at x=0, y=0 (bottom)
        # draw small rectangle for orifice
        ax_anim.add_patch(plt.Rectangle((-0.03, -0.05), 0.06, 0.05, color='saddlebrown'))
        # draw jet as a triangle whose width proportional to Q (visual)
        jet_length = 0.5
        jet_w = min(0.2, 0.02 + 10*qq)  # scale for visualization
        ax_anim.fill_between([ -jet_w/2, jet_w/2 ], [-0.05, -0.05], [-0.05-jet_length, -0.05-jet_length], color='cyan', alpha=0.6)

        # annotate numeric values
        ax_anim.text(0.02, 0.98*ax_anim.get_ylim()[1], f"t = {ti:.2f} s", verticalalignment='top', transform=ax_anim.transData)
        ax_anim.text(0.02, 0.92*ax_anim.get_ylim()[1], f"h = {hh:.4f} m", verticalalignment='top', transform=ax_anim.transData)
        ax_anim.text(0.02, 0.86*ax_anim.get_ylim()[1], f"Q = {qq:.6f} m³/s", verticalalignment='top', transform=ax_anim.transData)

        # --- plots: level vs time, Q vs time, Epot vs time ---
        fig_plots, axes = plt.subplots(3,1, figsize=(6,6))
        # level vs time
        axes[0].plot(t_phys, h_phys, '-k', linewidth=1)
        axes[0].scatter([ti], [hh], color='red')
        axes[0].set_ylabel("h (m)")
        axes[0].grid(True)
        # Q vs time
        axes[1].plot(t_phys, Q_phys, '-b', linewidth=1)
        axes[1].scatter([ti], [qq], color='red')
        axes[1].set_ylabel("Q (m³/s)")
        axes[1].grid(True)
        # Epot vs time
        axes[2].plot(t_phys, Epot, '-g', linewidth=1)
        axes[2].scatter([ti], [ee], color='red')
        axes[2].set_ylabel("E_p (J)")
        axes[2].set_xlabel("t (s)")
        axes[2].grid(True)
        fig_plots.tight_layout()

        # render figures
        anim_placeholder.pyplot(fig_anim)
        plots_place.pyplot(fig_plots)

        # small sleep to control FPS
        time.sleep(1.0/ fps)

    st.success("Simulación finalizada (tiempo físico mostrado hasta vaciado o máximo integrado).")
